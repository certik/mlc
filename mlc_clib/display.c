//
// Created by Brian Beckman on 3/12/24.
//

#include <stdio.h>

#include "display.h"


static const char *shades = " .:+#"; // " ░▒▓█"; // No Unicode?
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
                printf("%c%c", shades[4], shades[4]);
            } else if (v > 0.75) {
                printf("%c%c", shades[3], shades[3]);
            } else if (v > 0.50) {
                printf("%c%c", shades[2], shades[2]);
            } else if (v > 0.25) {
                printf("%c%c", shades[1], shades[1]);
            } else {
                printf("%c%c", shades[0], shades[0]);
            }
        }
        printf("%s\n", boxsid);
    }
    printf("%s\n", boxbtm);
}
