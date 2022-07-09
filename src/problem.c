#include <stdlib.h>
#include <stdio.h>

#include <math.h>
#include <float.h>
#include <errno.h>
#include <unistd.h>

#include "alloc.h"
#include "data_set.h"
#include "readline.h"
#include "rng.h"

#define DATA_SET_TEMPLATE "./%s/data/%s"
#define SPLIT_TEMPLATE    "./%s/splits/%s"

typedef struct data_set *(*generator)(uint32_t *, uint32_t **,
                                      uint32_t *, uint32_t **,
                                      double,
                                      rng_state);

static struct data_set *create_shell(uint16_t m, uint32_t n_leave_in, uint32_t n_hold_out,
                                     uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                     uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr)
{
    uint32_t i, n;
    uint16_t j, nchar;
    struct data_set *ds;

    *n_leave_in_ptr = n_leave_in;
    *leave_in_ptr = MALLOC(n_leave_in, sizeof(uint32_t));
    *n_hold_out_ptr = n_hold_out;
    *hold_out_ptr = MALLOC(n_hold_out, sizeof(uint32_t));

    nchar = ceil(log10(m + 1));

    ds = empty_data_set();

    ds->ninst = n_leave_in + n_hold_out;
    ds->nfeat = m;
    ds->var = MALLOC(m + 1, sizeof(struct variable));

    for (j = 0; j < m; ++j) {
        ds->var[j].name = MALLOC(nchar + 2, sizeof(char));
        sprintf(ds->var[j].name, "x%d", j);
        ds->var[j].type = CONTINUOUS;
        ds->var[j].levels = 0;
        ds->var[j].labels = NULL;
    }
    ds->var[m].name = strdup("y");
    ds->var[m].type = CONTINUOUS;
    ds->var[m].levels = 0;
    ds->var[m].labels = NULL;

    ds->X = MALLOC(ds->nfeat, sizeof(union data_point *));
    for (j = 0; j < ds->nfeat; ++j) {
        ds->X[j] = MALLOC(ds->ninst, sizeof(union data_point));
    }

    ds->y = MALLOC(ds->ninst, sizeof(union data_point));

    n = 0;
    for (i = 0; i < n_leave_in; ++i) (*leave_in_ptr)[i] = n++;
    for (i = 0; i < n_hold_out; ++i) (*hold_out_ptr)[i] = n++;

    return ds;
}

static struct data_set *generate_keijzer_1_3(double range,
                                             uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                             uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                             double loc)
{
    uint32_t i, n;
    double d;
    struct data_set *ds;

    ds = create_shell(1, range * 20 + 1, range * 2000 + 1,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        d = ds->X[0][n].f = -range + i * 0.1 + loc;
        ds->y[n].f = 0.3 * (d - loc) * sin(2 * M_PI * (d - loc));
    }

    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        d = ds->X[0][n].f = -range + i * 0.001 + loc;
        ds->y[n].f = 0.3 * (d - loc) * sin(2 * M_PI * (d - loc));
    }

    return ds;
}

static struct data_set *generate_keijzer_1(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    return generate_keijzer_1_3(1, n_leave_in_ptr, leave_in_ptr, n_hold_out_ptr, hold_out_ptr, loc);
}

static struct data_set *generate_keijzer_2(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    return generate_keijzer_1_3(2, n_leave_in_ptr, leave_in_ptr, n_hold_out_ptr, hold_out_ptr, loc);
}

static struct data_set *generate_keijzer_3(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    return generate_keijzer_1_3(3, n_leave_in_ptr, leave_in_ptr, n_hold_out_ptr, hold_out_ptr, loc);
}

static struct data_set *generate_keijzer_4(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    uint32_t i, n;
    double d;
    struct data_set *ds;

    ds = create_shell(1, 201, 201,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 0.0 + i * 0.05;
        ds->y[n].f  = d*d*d * exp(-d) * cos(d) * sin(d) * (sin(d)*sin(d) * cos(d) - 1);
        ds->X[0][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 0.05 + i * 0.05;
        ds->y[n].f  = d*d*d * exp(-d) * cos(d) * sin(d) * (sin(d)*sin(d) * cos(d) - 1);
        ds->X[0][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_5(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           rng_state state)
{
    uint32_t i, n;
    double x, y, z;
    struct data_set *ds;

    ds = create_shell(3, 1000, 10000,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -1 + 2 * next_rnd(state);
        y = ds->X[1][n].f =  1 + next_rnd(state);
        z = ds->X[2][n].f = -1 + 2 * next_rnd(state);
        ds->y[n].f  = (30 * x * z) / ((x - 10) * y*y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
        ds->X[2][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -1 + 2 * next_rnd(state);
        y = ds->X[1][n].f =  1 + next_rnd(state);
        z = ds->X[2][n].f = -1 + 2 * next_rnd(state);
        ds->y[n].f  = (30 * x * z) / ((x - 10) * y*y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
        ds->X[2][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_6(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    uint32_t i, j, n;
    double d;
    struct data_set *ds;

    ds = create_shell(1, 50, 120,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        d = 0;
        for (j = 1; j <= (i + 1); ++j) d += 1.0 / (double)j;
        ds->X[0][n].f = i + 1;
        ds->y[n].f = d;
        ds->X[0][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        d = 0;
        for (j = 1; j <= (i + 1); ++j) d += 1.0 / (double)j;
        ds->X[0][n].f = i + 1;
        ds->y[n].f = d;
        ds->X[0][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_7(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    uint32_t i, n;
    double d;
    struct data_set *ds;

    ds = create_shell(1, 100, 991,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);


    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 1 + i;
        ds->y[n].f = log(d);
        ds->X[0][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 1 + i * 0.1;
        ds->y[n].f = log(d);
        ds->X[0][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_8(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    uint32_t i, n;
    double d;
    struct data_set *ds;

    ds = create_shell(1, 101, 1001,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 1 + i;
        ds->y[n].f  = sqrt(d);
        ds->X[0][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 1 + i * 0.1;
        ds->y[n].f  = sqrt(d);
        ds->X[0][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_9(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                           uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                           double loc,
                                           __attribute__((unused)) rng_state state)
{
    uint32_t i, n;
    double d;
    struct data_set *ds;

    ds = create_shell(1, 101, 1001,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 1 + i;
        ds->y[n].f  = asinh(d);
        ds->X[0][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        d = ds->X[0][n].f = 1 + i * 0.1;
        ds->y[n].f  = asinh(d);
        ds->X[0][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_10(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i, j, k, n;
    double x, y;
    struct data_set *ds;

    ds = create_shell(2, 100, 10201,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x = ds->X[0][n].f = next_rnd(state);
        y = ds->X[1][n].f = next_rnd(state);
        ds->y[n].f  = pow(x, y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        j = i / 101;
        k = i % 101;
        x = ds->X[0][n].f = j * 0.01;
        y = ds->X[1][n].f = k * 0.01;
        ds->y[n].f  = pow(x, y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_11(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i, j, k, n;
    double x, y;
    struct data_set *ds;

    ds = create_shell(2, 20, 361201,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -3 + 6 * next_rnd(state);
        y = ds->X[1][n].f = -3 + 6 * next_rnd(state);
        ds->y[n].f  = x * y + sin((x - 1) * (y - 1));
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        j = i / 601;
        k = i % 601;
        x = ds->X[0][n].f = -3 + j * 0.01;
        y = ds->X[1][n].f = -3 + k * 0.01;
        ds->y[n].f  = x * y + sin((x - 1) * (y - 1));
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_12(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i, j, k, n;
    double x, y;
    struct data_set *ds;

    ds = create_shell(2, 20, 361201,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -3 + 6 * next_rnd(state);
        y = ds->X[1][n].f = -3 + 6 * next_rnd(state);
        ds->y[n].f  = x*x*x*x - x*x*x + y*y/2 - y;
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        j = i / 601;
        k = i % 601;
        x = ds->X[0][n].f = -3 + j * 0.01;
        y = ds->X[1][n].f = -3 + k * 0.01;
        ds->y[n].f  = x*x*x*x - x*x*x + y*y/2 - y;
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_13(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i, j, k, n;
    double x, y;
    struct data_set *ds;

    ds = create_shell(2, 20, 361201,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -3 + 6 * next_rnd(state);
        y = ds->X[1][n].f = -3 + 6 * next_rnd(state);
        ds->y[n].f  = 6 * sin(x) * cos(y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        j = i / 601;
        k = i % 601;
        x = ds->X[0][n].f = -3 + j * 0.01;
        y = ds->X[1][n].f = -3 + k * 0.01;
        ds->y[n].f  = 6 * sin(x) * cos(y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_14(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i, j, k, n;
    double x, y;
    struct data_set *ds;

    ds = create_shell(2, 20, 361201,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -3 + 6 * next_rnd(state);
        y = ds->X[1][n].f = -3 + 6 * next_rnd(state);
        ds->y[n].f  = 8 / (2 + x*x + y*y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        j = i / 601;
        k = i % 601;
        x = ds->X[0][n].f = -3 + j * 0.01;
        y = ds->X[1][n].f = -3 + k * 0.01;
        ds->y[n].f  = 8 / (2 + x*x + y*y);
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_keijzer_15(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i, j, k, n;
    double x, y;
    struct data_set *ds;

    ds = create_shell(2, 20, 361201,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -3 + 6 * next_rnd(state);
        y = ds->X[1][n].f = -3 + 6 * next_rnd(state);
        ds->y[n].f  = x*x*x/5 + y*y*y/2 - y - x;
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        j = i / 601;
        k = i % 601;
        x = ds->X[0][n].f = -3 + j * 0.01;
        y = ds->X[1][n].f = -3 + k * 0.01;
        ds->y[n].f  = x*x*x/5 + y*y*y/2 - y - x;
        ds->X[0][n].f += loc;
        ds->X[1][n].f += loc;
    }

    return ds;
}

static struct data_set *generate_friedman_1(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i;
    double x0, x1, x2, x3, x4;
    struct data_set *ds;

    ds = create_shell(10, 200, 2000,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (i = 0; i < (*n_leave_in_ptr + *n_hold_out_ptr); ++i) {
        x0 = ds->X[0][i].f = next_rnd(state);
        x1 = ds->X[1][i].f = next_rnd(state);
        x2 = ds->X[2][i].f = next_rnd(state);
        x3 = ds->X[3][i].f = next_rnd(state);
        x4 = ds->X[4][i].f = next_rnd(state);
        ds->X[5][i].f = next_rnd(state); /* noise variable - not used */
        ds->X[6][i].f = next_rnd(state); /* noise variable - not used */
        ds->X[7][i].f = next_rnd(state); /* noise variable - not used */
        ds->X[8][i].f = next_rnd(state); /* noise variable - not used */
        ds->X[9][i].f = next_rnd(state); /* noise variable - not used */
        ds->y[i].f  = 10 * sin(M_PI * x0 * x1) + 20 * (x2 - 0.5) * (x2 - 0.5) + 10 * x3 + 5 * x4 + next_rnd_gauss(0, 1, state);
        ds->X[0][i].f += loc;
        ds->X[1][i].f += loc;
        ds->X[2][i].f += loc;
        ds->X[3][i].f += loc;
        ds->X[4][i].f += loc;
        ds->X[5][i].f += loc;
        ds->X[6][i].f += loc;
        ds->X[7][i].f += loc;
        ds->X[8][i].f += loc;
        ds->X[9][i].f += loc;
    }

    return ds;
}

static struct data_set *generate_friedman_2(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i;
    double R, f, L, C, Z;
    double delta, meanZ, m2Z, sdZ;
    struct data_set *ds;

    ds = create_shell(4, 200, 2000,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    meanZ = m2Z = 0;
    for (i = 0; i < (*n_leave_in_ptr + *n_hold_out_ptr); ++i) {
        R = ds->X[0][i].f = next_rnd(state) * 100;
        f = ds->X[1][i].f = 20 + next_rnd(state) * 260;
        L = ds->X[2][i].f = next_rnd(state);
        C = ds->X[3][i].f = 1 + next_rnd(state) * 10;

        Z = ds->y[i].f = sqrt(R*R + (2*M_PI*f*L - 1 / (2*M_PI*f*C))*(2*M_PI*f*L - 1 / (2*M_PI*f*C)));

        delta = Z - meanZ;
        meanZ += delta / (i + 1);
        m2Z += delta * (Z - meanZ);

        ds->X[0][i].f += loc;
        ds->X[1][i].f += loc;
        ds->X[2][i].f += loc;
        ds->X[3][i].f += loc;
    }

    sdZ = sqrt(m2Z / (ds->ninst - 1));
    for (i = 0; i < (*n_leave_in_ptr + *n_hold_out_ptr); ++i) ds->y[i].f += next_rnd_gauss(0, sdZ, state) / 3; /* SNR of 3 */

    return ds;
}

static struct data_set *generate_friedman_3(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                            uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                            double loc,
                                            rng_state state)
{
    uint32_t i;
    double R, f, L, C, Z;
    double delta, meanZ, m2Z, sdZ;
    struct data_set *ds;

    ds = create_shell(4, 200, 2000,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    meanZ = m2Z = 0;
    for (i = 0; i < (*n_leave_in_ptr + *n_hold_out_ptr); ++i) {
        R = ds->X[0][i].f = next_rnd(state) * 100;
        f = ds->X[1][i].f = 20 + next_rnd(state) * 260;
        L = ds->X[2][i].f = next_rnd(state);
        C = ds->X[3][i].f = 1 + next_rnd(state) * 10;

        Z = ds->y[i].f = atan((2*M_PI*f*L - 1 / (2*M_PI*f*C)) / R);;

        delta = Z - meanZ;
        meanZ += delta / (i + 1);
        m2Z += delta * (Z - meanZ);

        ds->X[0][i].f += loc;
        ds->X[1][i].f += loc;
        ds->X[2][i].f += loc;
        ds->X[3][i].f += loc;
    }

    sdZ = sqrt(m2Z / (ds->ninst - 1));
    for (i = 0; i < (*n_leave_in_ptr + *n_hold_out_ptr); ++i) ds->y[i].f += next_rnd_gauss(0, sdZ, state) / 3; /* SNR of 3 */

    return ds;
}

static struct data_set *generate_pagie_1(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                         uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                         double loc,
                                         rng_state state)
{
    uint32_t i, j, k, n;
    double x, y;
    struct data_set *ds;

    ds = create_shell(2, 676, 1000,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        j = i / 26;
        k = i % 26;
        x = ds->X[0][n].f = -5 + 0.4 * j;
        y = ds->X[1][n].f = -5 + 0.4 * k;
        ds->y[n].f  = 1 / (1 + pow(x, -4)) + 1 / (1 + pow(y, -4));
        ds->X[0][i].f += loc;
        ds->X[1][i].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        x = ds->X[0][n].f = -5 + next_rnd(state) * 10;
        y = ds->X[1][n].f = -5 + next_rnd(state) * 10;
        ds->y[n].f  = 1 / (1 + pow(x, -4)) + 1 / (1 + pow(y, -4));
        ds->X[0][i].f += loc;
        ds->X[1][i].f += loc;
    }

    return ds;
}

static struct data_set *generate_vladislavleva_14(uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                                  uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                                  double loc,
                                                  rng_state state)
{
    uint32_t i, n;
    double x0, x1, x2, x3, x4;
    struct data_set *ds;

    ds = create_shell(5, 1024, 5000,
                      n_leave_in_ptr, leave_in_ptr,
                      n_hold_out_ptr, hold_out_ptr);

    for (n = 0, i = 0; i < *n_leave_in_ptr; ++i, ++n) {
        x0 = ds->X[0][n].f = 0.05 + 6 * next_rnd(state);
        x1 = ds->X[1][n].f = 0.05 + 6 * next_rnd(state);
        x2 = ds->X[2][n].f = 0.05 + 6 * next_rnd(state);
        x3 = ds->X[3][n].f = 0.05 + 6 * next_rnd(state);
        x4 = ds->X[4][n].f = 0.05 + 6 * next_rnd(state);

        ds->y[n].f  = 10 / (5 + (x0 - 3)*(x0 - 3)+ (x1 - 3)*(x1 - 3)+ (x2 - 3)*(x2 - 3)+ (x3 - 3)*(x3 - 3)+ (x4 - 3)*(x4 - 3));

        ds->X[0][i].f += loc;
        ds->X[1][i].f += loc;
        ds->X[2][i].f += loc;
        ds->X[3][i].f += loc;
        ds->X[4][i].f += loc;
    }
    for (i = 0; i < *n_hold_out_ptr; ++i, ++n) {
        x0 = ds->X[0][n].f = -0.25 + 6.6 * next_rnd(state);
        x1 = ds->X[1][n].f = -0.25 + 6.6 * next_rnd(state);
        x2 = ds->X[2][n].f = -0.25 + 6.6 * next_rnd(state);
        x3 = ds->X[3][n].f = -0.25 + 6.6 * next_rnd(state);
        x4 = ds->X[4][n].f = -0.25 + 6.6 * next_rnd(state);

        ds->y[n].f  = 10 / (5 + (x0 - 3)*(x0 - 3)+ (x1 - 3)*(x1 - 3)+ (x2 - 3)*(x2 - 3)+ (x3 - 3)*(x3 - 3)+ (x4 - 3)*(x4 - 3));

        ds->X[0][i].f += loc;
        ds->X[1][i].f += loc;
        ds->X[2][i].f += loc;
        ds->X[3][i].f += loc;
        ds->X[4][i].f += loc;
    }

    return ds;
}

static void sample_split(uint32_t ninst, double train_frac,
                         uint32_t **leave_in_ptr, uint32_t *n_leave_in_ptr,
                         uint32_t **hold_out_ptr, uint32_t *n_hold_out_ptr,
                         rng_state state)
{
    uint32_t i, N, *leave_in, n_leave_in, *hold_out, n_hold_out;

    /* sample a train and test fraction */
    N = train_frac * ninst;
    leave_in = MALLOC(ninst, sizeof(uint32_t));
    hold_out = MALLOC(ninst, sizeof(uint32_t));
    for (n_leave_in = n_hold_out = 0, i = 0; i < ninst && n_leave_in < N; ++i) {
        if (next_rnd(state) < ((double)(N - n_leave_in) / (double)(ninst - i))) {
            leave_in[n_leave_in++] = i;
        } else {
            hold_out[n_hold_out++] = i;
        }
    }
    while (i < ninst) hold_out[n_hold_out++] = i++;

    *leave_in_ptr = leave_in;
    *hold_out_ptr = hold_out;
    *n_leave_in_ptr = n_leave_in;
    *n_hold_out_ptr = n_hold_out;
}



static void load_split(const char *name, const char *problem_dir, uint16_t split, uint32_t ninst,
                       uint32_t *leave_in, uint32_t *n_leave_in_ptr,
                       uint32_t *hold_out, uint32_t *n_hold_out_ptr)
{
    uint32_t i;

    FILE *data;
    char *split_file;
    char *buffer = NULL, *line = NULL;
    size_t bufsz = 0;

    uint32_t n_leave_in, n_hold_out;

    split_file = MALLOC(strlen(SPLIT_TEMPLATE) + strlen(name) + strlen(problem_dir), sizeof(char));
    sprintf(split_file, SPLIT_TEMPLATE, problem_dir, name);

    data = fopen(split_file, "r");
    if (data == NULL) {
        fprintf(stderr, "%s:%d - ERROR: Train/Test split #%u from file %s could not be loaded. Reason (%d) %s\n",
                __FILE__, __LINE__, split, split_file, errno, strerror(errno));
        exit(EXIT_FAILURE);
    } else {
        for (i = 0; i < split; ++i) line = next_line(&buffer, &bufsz, data);
        fclose(data);

        n_leave_in = n_hold_out = 0;
        for (i = 0; i < ninst; ++i, ++line) {
            if (*line == '0') leave_in[n_leave_in++] = i;
            else if (*line == '1') hold_out[n_hold_out++] = i;
        }
    }

    free(split_file);
    free(buffer);

    *n_leave_in_ptr = n_leave_in;
    *n_hold_out_ptr = n_hold_out;
}



struct data_set *load_problem(const char *name, const char *problem_dir, double train_frac,
                              uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                              uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                              rng_state state)
{
    static char *fn_name[] = {
        "F1", "F2", "F3",
        "K01", "K02", "K03", "K04", "K05", "K06", "K07", "K08", "K09", "K10", "K11", "K12", "K13", "K14", "K15",
        "P1",
        "V14",
        NULL
    };

    static generator fn_generator[] = {
        generate_friedman_1, generate_friedman_2, generate_friedman_3,
        generate_keijzer_1, generate_keijzer_2, generate_keijzer_3, generate_keijzer_4, generate_keijzer_5, generate_keijzer_6, generate_keijzer_7, generate_keijzer_8, generate_keijzer_9, generate_keijzer_10,
        generate_keijzer_11, generate_keijzer_12, generate_keijzer_13, generate_keijzer_14, generate_keijzer_15,
        generate_pagie_1,
        generate_vladislavleva_14
    };

    char *data_file;
    struct data_set *ds = NULL;
    uint32_t *leave_in = NULL, n_leave_in, *hold_out = NULL, n_hold_out;
    char *offset = NULL;
    double loc = 0;

    uint8_t fn = 0;

    if (name == NULL) {
        fprintf(stderr, "%s:%d - WARNING: No problem name supplied!\n",
                __FILE__, __LINE__);
        return NULL;
    }

    data_file = MALLOC(strlen(DATA_SET_TEMPLATE) + strlen(name) + strlen(problem_dir), sizeof(char));
    sprintf(data_file, DATA_SET_TEMPLATE, problem_dir, name);

    if (access(data_file, F_OK) != -1) {
        /* load the file */
        ds = load_data(data_file);


        if (train_frac == -1) { /* special case - no validation, use entire data set for training */
            n_leave_in = ds->ninst;
            n_hold_out = 0;
            leave_in = MALLOC(ds->ninst, sizeof(uint32_t));
            hold_out = MALLOC(ds->ninst, sizeof(uint32_t));
            for (uint32_t i = 0; i < ds->ninst; ++i) leave_in[i] = i;
        } else if (train_frac < 0) {
            n_leave_in = ds->ninst;
            n_hold_out = 0;
            fprintf(stderr, "%s:%d - WARNING: Invalid training fraction of %f supplied!\n",
                    __FILE__, __LINE__, train_frac);
        } else if (train_frac < 1.0) {
            sample_split(ds->ninst, train_frac, &leave_in, &n_leave_in, &hold_out, &n_hold_out, state);
        } else {
            /* load the split */
            leave_in = malloc(ds->ninst * sizeof(uint32_t));
            hold_out = malloc(ds->ninst * sizeof(uint32_t));
            load_split(name, problem_dir, train_frac, ds->ninst,
                       leave_in, &n_leave_in, hold_out, &n_hold_out);
        }

        *n_leave_in_ptr = n_leave_in;
        *n_hold_out_ptr = n_hold_out;
        *leave_in_ptr = leave_in;
        *hold_out_ptr = hold_out;
    } else {
        if ((offset = strchr(name, '_')) != NULL) loc = atof(offset + 1);

        while (fn_name[fn]) {
            if (strncmp(name, fn_name[fn], strlen(fn_name[fn])) == 0) {
                ds = fn_generator[fn](n_leave_in_ptr, leave_in_ptr, n_hold_out_ptr, hold_out_ptr, loc, state);
                break;
            }
            fn++;
        }

        if (ds == NULL) {
            fprintf(stderr, "%s:%d - WARNING: Unknown problem name \"%s\"\n",
                    __FILE__, __LINE__, name);
        }
    }

    free(data_file);

    return ds;
}
