//
// Created by Brian Beckman on 3/12/24.
//

#include <stdio.h>

#include "display.h"


static const char *shade0 = " ";
static const char *shade1 = "░";
static const char *shade2 = "▒";
static const char *shade3 = "▓";
static const char *shade4 = "█";
static const char *boxtop = "╔════════════════════════════════════════════════════════╗";
static const char *boxbtm = "╚════════════════════════════════════════════════════════╝";
static const char *boxsid = "║";

void draw_digit(const f32 *p28x28) {
    printf("%s\n", boxtop);
    for (int row = 0; row < 28; row++) {
        printf("%s", boxsid);
        for (int col = 0; col < 28; col++) {
            f32 v = p28x28[col + (28 * row)];
            if (v > 0.99) {
                printf("%s%s", shade4, shade4);
            } else if (v > 0.75) {
                printf("%s%s", shade3, shade3);
            } else if (v > 0.50) {
                printf("%s%s", shade2, shade2);
            } else if (v > 0.25) {
                printf("%s%s", shade1, shade1);
            } else {
                printf("%s%s", shade0, shade0);
            }
        }
        printf("%s\n", boxsid);
    }
    printf("%s\n", boxbtm);
}
