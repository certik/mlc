//
// Created by Brian Beckman on 3/12/24.
//

#include "gguf.h"

int ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}

const char * ggml_type_name(enum ggml_type type) {
    return type_traits[type].type_name;
}

static bool gguf_fread_el(FILE * file, void * dst, size_t size,
                          size_t * offset)
{
    const size_t n = fread(dst, 1, size, file);
    *offset += n;
    return n == size;
}

static bool gguf_fread_str(FILE * file, struct gguf_str * p, size_t * offset) {
    p->n    = 0;
    p->data = NULL;
    bool ok = true;
    ok = ok && gguf_fread_el(file, &p->n, sizeof(p->n), offset);
    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%llu)\n", __func__, p->n);
        return false;
    }
    p->data = calloc(p->n + 1, 1);
    ok = ok && gguf_fread_el(file,  p->data, p->n, offset);
    return ok;
}



int gguf_read(const char *fname, struct gguf_context *ctx)
{
    FILE * file = fopen(fname, "rb");
    if (!file) {
        fprintf(stderr, "%s: failed to open `%s`\n", __func__, fname);
        return 1;
    }
    size_t offset = 0;

    // Magic
    {
        char magic[4];
        gguf_fread_el(file, &magic, sizeof(magic), &offset);
        for (uint32_t i = 0; i < sizeof(magic); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters '%c%c%c%c'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                fclose(file);
                return 2;
            }
        }
        strncpy(ctx->header.magic, magic, 4);
    }

    // Header
    {
        bool ok = true;

        ctx->kv    = NULL;
        ctx->infos = NULL;
        ctx->data  = NULL;

        ok = ok && gguf_fread_el(file, &ctx->header.version,
                                 sizeof(ctx->header.version),   &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_tensors,
                                 sizeof(ctx->header.n_tensors), &offset);
        ok = ok && gguf_fread_el(file, &ctx->header.n_kv,
                                 sizeof(ctx->header.n_kv),      &offset);

        if (ctx->header.version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            return 3;
        }

        // sanity-checks to prevent from integer/buffer overflows
        ok = ok && (ctx->header.n_tensors
                    < (SIZE_MAX/2)/sizeof(struct gguf_tensor_info));
        ok = ok && (ctx->header.n_kv
                    < (SIZE_MAX/2)/sizeof(struct gguf_kv));

        if (!ok) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            return 4;
        }
    }

    // kv pairs
    {
        bool ok = true;

        ctx->kv = malloc(ctx->header.n_kv * sizeof(struct gguf_kv));

        for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

            //fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            ok = ok && gguf_fread_str(file, &kv->key,                    &offset);
            ok = ok && gguf_fread_el (file, &kv->type, sizeof(kv->type), &offset);

            //fprintf(stderr, "%s: reading kv with key %s\n", __func__, kv->key.data);

            switch (kv->type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_fread_el (file, &kv->value.uint8,   sizeof(kv->value.uint8),   &offset); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_fread_el (file, &kv->value.int8,    sizeof(kv->value.int8),    &offset); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_fread_el (file, &kv->value.uint16,  sizeof(kv->value.uint16),  &offset); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_fread_el (file, &kv->value.int16,   sizeof(kv->value.int16),   &offset); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_fread_el (file, &kv->value.uint32,  sizeof(kv->value.uint32),  &offset); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_fread_el (file, &kv->value.int32,   sizeof(kv->value.int32),   &offset); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_fread_el (file, &kv->value.float32, sizeof(kv->value.float32), &offset); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_fread_el (file, &kv->value.uint64,  sizeof(kv->value.uint64),  &offset); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_fread_el (file, &kv->value.int64,   sizeof(kv->value.int64),   &offset); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_fread_el (file, &kv->value.float64, sizeof(kv->value.float64), &offset); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_fread_el (file, &kv->value.bool_,   sizeof(kv->value.bool_),   &offset); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_fread_str(file, &kv->value.str,                                &offset); break;
                case GGUF_TYPE_ARRAY:
                {
                    ok = ok && gguf_fread_el(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
                    ok = ok && gguf_fread_el(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);

                    switch (kv->value.arr.type) {
                        case GGUF_TYPE_UINT8:
                        case GGUF_TYPE_INT8:
                        case GGUF_TYPE_UINT16:
                        case GGUF_TYPE_INT16:
                        case GGUF_TYPE_UINT32:
                        case GGUF_TYPE_INT32:
                        case GGUF_TYPE_FLOAT32:
                        case GGUF_TYPE_UINT64:
                        case GGUF_TYPE_INT64:
                        case GGUF_TYPE_FLOAT64:
                        case GGUF_TYPE_BOOL:
                        {
                            // prevent from integer overflow in the malloc below
                            if (kv->value.arr.n >= SIZE_MAX/gguf_type_size(kv->value.arr.type)) {
                                fprintf(stderr, "%s: array size is too large (%llu)\n", __func__, kv->value.arr.n);
                                fclose(file);
                                return 5;
                            }

                            kv->value.arr.data = malloc(kv->value.arr.n * gguf_type_size(kv->value.arr.type));

                            ok = ok && gguf_fread_el(file, kv->value.arr.data, kv->value.arr.n * gguf_type_size(kv->value.arr.type), &offset);
                        } break;
                        case GGUF_TYPE_STRING:
                        {
                            // prevent from integer overflow in the malloc below
                            if (kv->value.arr.n >= SIZE_MAX/sizeof(struct gguf_str)) {
                                fprintf(stderr, "%s: array size is too large (%llu)\n", __func__, kv->value.arr.n);
                                fclose(file);
                                return 6;
                            }

                            kv->value.arr.data = malloc(kv->value.arr.n * sizeof(struct gguf_str));

                            for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
                                ok = ok && gguf_fread_str(file, &((struct gguf_str *) kv->value.arr.data)[j], &offset);
                            }
                        } break;
                        case GGUF_TYPE_ARRAY:
                        default: fprintf(stderr, "invalid type\n"); return 8; break;
                    }
                } break;
                default: fprintf(stderr, "invalid type\n"); return 9; break;
            }

            if (!ok) {
                break;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            return 7;
        }
    }

    // read the array infos
    {
        bool ok = true;

        ctx->infos = malloc(ctx->header.n_tensors * sizeof(struct gguf_tensor_info));

        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < GGUF_MAX_DIMS; ++j) {
                info->ne[j] = 1;
            }

            ok = ok && gguf_fread_str(file, &info->name,                          &offset);
            ok = ok && gguf_fread_el (file, &info->n_dims, sizeof(info->n_dims),  &offset);

            ok = ok && (info->n_dims <= GGUF_MAX_DIMS);

            for (uint32_t j = 0; j < info->n_dims; ++j) {
                ok = ok && gguf_fread_el(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }

            ok = ok && gguf_fread_el (file, &info->type,   sizeof(info->type),    &offset);
            ok = ok && gguf_fread_el (file, &info->offset, sizeof(info->offset),  &offset);

            //gguf_tensor_info_sanitize(info);

            if (!ok) {
                fprintf(stderr, "%s: failed to read array info\n", __func__);
                fclose(file);
                return 10;
            }
        }
    }

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    // Not needed for now
    /*
    int alignment_idx = gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
    }
    */

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;

        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseek(file, offset, SEEK_SET);
        }
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = offset;

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            const int64_t ne =
                    (int64_t) info->ne[0] *
                    (int64_t) info->ne[1] *
                    (int64_t) info->ne[2] *
                    (int64_t) info->ne[3];

            if (ne % ggml_blck_size(info->type) != 0) {
                fprintf(stderr, "%s: array '%s' of type %d (%s) number of elements (%llu) is not a multiple of block size (%d)\n",
                        __func__, info->name.data, (int)info->type, ggml_type_name(info->type), ne, ggml_blck_size(info->type));
                fclose(file);
                return 11;
            }

            const size_t size_cur = ggml_row_size(info->type, ne);

            ctx->size += GGUF_PAD(size_cur, ctx->alignment);
        }
    }

    // Read Data
    {
        bool ok = true;
        ctx->data = malloc(ctx->size);
        ok = ok && gguf_fread_el(file, ctx->data, ctx->size, &offset);
        if (!ok) {
            fprintf(stderr, "%s: failed to read array data\n", __func__);
            fclose(file);
            return 12;
        }
    }

    return 0;
}

