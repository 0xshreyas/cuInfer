#include <stdio.h>

int main() {
    float big = 1e8f;
    printf("1e8 + 1  = %.1f\n", big + 1.0f);

    int n = 10000000;
    float fwd = 0.0f, bwd = 0.0f;
    for (int i = 1; i <= n; i++) fwd += 1.0f / i;
    for (int i = n; i >= 1; i--) bwd += 1.0f / i;
    printf("sum forward:  %.7f\n", fwd);
    printf("sum backward: %.7f\n", bwd);
    return 0;
}
