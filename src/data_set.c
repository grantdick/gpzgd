#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>

#include <stdbool.h>
#include <stdint.h>

#include <errno.h>
#include <math.h>
#include <string.h>

#include "alloc.h"
#include "data_set.h"
#include "readline.h"

struct variable *copy_variable_info(struct variable *var, uint16_t nvar)
{
    struct variable *copy;

    uint16_t l, j;

    if (var == NULL || nvar <= 0) return NULL;

    copy  = MALLOC(nvar, sizeof(struct variable));

    for (j = 0; j < nvar; ++j) {
        copy[j].type = var[j].type;
        copy[j].levels = var[j].levels;

        if (var[j].name) {
            copy[j].name = MALLOC(strlen(var[j].name) + 1, sizeof(char));
            strcpy(copy[j].name, var[j].name);
        } else {
            var[j].name = NULL;
        }

        if (var[j].labels) {
            copy[j].labels = MALLOC(var[j].levels, sizeof(char *));
            for (l = 0; l < var[j].levels; ++l) {
                copy[j].labels[l] = MALLOC(strlen(var[j].labels[l]) + 1, sizeof(char));
                strcpy(copy[j].labels[l], var[j].labels[l]);
            }
        } else {
            copy[j].labels = NULL;
        }
    }

    return copy;
}

struct data_set *empty_data_set()
{
    struct data_set *ds;

    ds = MALLOC(1, sizeof(struct data_set));

    ds->ninst = 0;
    ds->nfeat = 0;

    ds->var = NULL;

    ds->X = NULL;
    ds->y = NULL;

    return ds;
}


void release_data_set(struct data_set *ds)
{
    uint16_t j;

    if (ds == NULL) return;

    release_variable_info(ds->var, ds->nfeat + 1);

    for (j = 0; j < ds->nfeat; ++j) free(ds->X[j]);
    free(ds->X);
    free(ds->y);

    free(ds);
}

void release_variable_info(struct variable *var, uint16_t nvar)
{
    uint16_t l, j;

    if (var) {
        for (j = 0; j < nvar; ++j) {
            free(var[j].name);
            if (var[j].labels) {
                for (l = 0; l < var[j].levels; ++l) free(var[j].labels[l]);
                free(var[j].labels);
            }
        }
        free(var);
    }
}

struct data_set *load_data(const char *source)
{
    uint32_t i;
    uint16_t j;
    uint16_t response_col;

    struct data_set *ds;
    struct variable var;
    FILE *data;
    char *buffer, *line, *tok;
    size_t bufsz;
    char col_type;

    ds = empty_data_set();

    data = fopen(source, "r");
    if (data == NULL) {
        fprintf(stderr,
                "ERROR (%s:%d): Problem opening data file. Reason: %d (%s). Quitting.\n",
                __FILE__, __LINE__,
                errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    buffer = line = tok = NULL;
    bufsz = 0;

    /* read the header */
    line = next_line(&buffer, &bufsz, data);
    if (line == NULL) {
        fprintf(stderr,
                "ERROR (%s:%d): Problem reading data file header. Quitting.\n",
                __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    if (sscanf(line, "%u %hu", &(ds->ninst), &(ds->nfeat)) != 2) {
        fprintf(stderr,
                "ERROR (%s:%d): Problem reading data file header. Reason: %d (%s). Quitting.\n",
                __FILE__, __LINE__,
                errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (ds->ninst > 0 && ds->nfeat > 0) {
        ds->var  = MALLOC(ds->nfeat, sizeof(struct variable));

        ds->X = MALLOC(ds->nfeat, sizeof(union data_point *));

        response_col = ds->nfeat - 1; /* assume that the response is the last column */
        for (j = 0; j < ds->nfeat; ++j) {
            ds->X[j] = MALLOC(ds->ninst, sizeof(union data_point));

            ds->var[j].name = MALLOC(256, sizeof(char));
            ds->var[j].labels = NULL;
            ds->var[j].levels = 0;

            line = next_line(&buffer, &bufsz, data);
            tok = strtok(line, " \t");
            strcpy(ds->var[j].name, tok);

            tok = strtok(NULL, " \t");
            col_type = (tok == NULL) ? 'C' : tok[0];
            if (col_type == 'R') {
                response_col = j;
                tok = strtok(NULL, " \t");
                col_type = (tok == NULL) ? 'C' : tok[0];
            }

            ds->var[j].type = col_type == 'C' ? CONTINUOUS : CATEGORICAL;
            if (ds->var[j].type == CATEGORICAL) {
                tok = strtok(NULL, " \t");
                while (strcmp(tok, "}") != 0) {
                    ds->var[j].labels = REALLOC(ds->var[j].labels, ds->var[j].levels + 1, sizeof(char *));
                    ds->var[j].labels[ds->var[j].levels] = MALLOC(256, sizeof(char));
                    strcpy(ds->var[j].labels[ds->var[j].levels], tok);

                    ds->var[j].levels++;

                    tok = strtok(NULL, " \t");
                }
            }
        }

        for (i = 0; i < ds->ninst; ++i) {
            line = next_line(&buffer, &bufsz, data);
            for (j = 0, tok = strtok(line, " \t"); j < ds->nfeat; ++j, tok = strtok(NULL, " \t")) {
                if (tok[0] == '?') {
                    ds->X[j][i] = unknown_value(ds->var[j].type);
                } else if (ds->var[j].type == CONTINUOUS) {
                    ds->X[j][i].f = atof(tok);
                } else {
                    ds->X[j][i].c = 0;
                    while (strcmp(ds->var[j].labels[ds->X[j][i].c], tok) != 0) ds->X[j][i].c++;
                }
            }
        }

        ds->y = ds->X[response_col];
        if (response_col != (ds->nfeat - 1)) {
            var = ds->var[response_col];

            memmove(ds->X + response_col, ds->X + response_col + 1,
                    (ds->nfeat - response_col - 1) * sizeof(double *));

            memmove(ds->var + response_col, ds->var + response_col + 1,
                    (ds->nfeat - response_col - 1) * sizeof(struct variable));

            ds->var[ds->nfeat - 1] = var;
            ds->X[ds->nfeat - 1] = ds->y; /* no reason for this, just "feels" tidier :) */
        }
        ds->nfeat--; /* account for the response column */
    }

    fclose(data);
    free(buffer);

    return ds;
}



void split_data(struct data_set *ds, uint8_t *partition,
                struct data_set **keepin_ptr, struct data_set **holdout_ptr)
{
    uint32_t i;
    uint16_t j;
    struct data_set *keepin, *holdout;

    *keepin_ptr = *holdout_ptr = NULL;
    if (ds == NULL) return;

    keepin = empty_data_set();
    holdout = empty_data_set();

    keepin->nfeat = holdout->nfeat = ds->nfeat;

    keepin->X = MALLOC(ds->nfeat, sizeof(double *));
    holdout->X = MALLOC(ds->nfeat, sizeof(double *));
    for (j = 0; j < ds->nfeat; ++j) {
        keepin->X[j] = MALLOC(ds->ninst, sizeof(double *));
        holdout->X[j] = MALLOC(ds->ninst, sizeof(double *));
    }

    keepin->y = MALLOC(ds->ninst, sizeof(double));
    holdout->y = MALLOC(ds->ninst, sizeof(double));

    keepin->var = copy_variable_info(ds->var, ds->nfeat + 1);
    holdout->var = copy_variable_info(ds->var, ds->nfeat + 1);

    keepin->ninst = holdout->ninst = 0;
    for (i = 0; i < ds->ninst; ++i) {
        if (partition[i] == 0) {
            for (j = 0; j < ds->nfeat; ++j) keepin->X[j][keepin->ninst] = ds->X[j][i];
            keepin->y[keepin->ninst++] = ds->y[i];
        } else if (partition[i] == 1) {
            for (j = 0; j < ds->nfeat; ++j) holdout->X[j][holdout->ninst] = ds->X[j][i];
            holdout->y[holdout->ninst++] = ds->y[i];
        } /* otherwise skip for anything that isn't in { 0, 1 } */
    }

    *keepin_ptr = keepin;
    *holdout_ptr = holdout;
}



/* simple comparer for the qsort_r function that sorts an integer
 * array based upon the values in a corresponding double array */
static int32_t cmp_idx_flt(const void *a_ptr, const void *b_ptr, void *data_ptr)
{
    union data_point *data;

    uint32_t a, b;

    a = *((const uint32_t * const)a_ptr);
    b = *((const uint32_t * const)b_ptr);
    data = data_ptr;

    if (isfinite(data[a].f) && isfinite(data[b].f)) { /* both values present */
        return (data[a].f < data[b].f) ? -1 : ((data[a].f > data[b].f) ? 1 : 0);
    } else if (isfinite(data[a].f)) { /* first present, second missing */
        return -1;
    } else if (isfinite(data[b].f)) { /* second missing, first present */
        return 1;
    } else { /* both values are missing */
        return 0;
    }
}
/* simple comparer for the qsort_r function that sorts an integer
 * array based upon the values in a corresponding double array */
static int32_t cmp_idx_cat(const void *a_ptr, const void *b_ptr, void *data_ptr)
{
    union data_point *data;

    uint32_t a, b;

    a = *((const uint32_t * const)a_ptr);
    b = *((const uint32_t * const)b_ptr);
    data = data_ptr;

    if ((data[a].c < 0) && (data[b].c < 0)) { /* both missing */
        return 0;
    } else if (data[a].c < 0) { /* first value missing */
        return 1;
    } else if (data[b].c < 0) { /* second value missing */
        return -1;
    } else { /* neither value missing, straight integer sort */
        return data[a].c - data[b].c;
    }
}
void sort_index(union data_point *x, uint32_t *idx, uint32_t nsamples, enum feature_type type)
{
    qsort_r(idx, nsamples, sizeof(int), (type == CONTINUOUS) ? cmp_idx_flt : cmp_idx_cat, x);
}



bool missing_value(union data_point x, enum feature_type type)
{
    return (type == CONTINUOUS) ? isnan(x.f) : (x.c < 0);
}



int32_t compare_values(union data_point x, union data_point y,
                       enum feature_type type, bool compl)
{
    if (type == CONTINUOUS) {
        if (isnan(x.f)) return 0;
        return (x.f <  y.f ? -1 : 1) * (compl ? -1 : 1);
    } else {
        if (x.c < 0) return 0;
        return ((1 << x.c) & y.c ? -1 : 1) * (compl ? -1 : 1);
    }
}



union data_point unknown_value(enum feature_type type)
{
    union data_point x;

    if (type == CONTINUOUS) x.f = NAN; else x.c = -1;

    return x;
}



uint16_t encoding_size(struct data_set *ds)
{
    uint16_t j, res;

    res = 0;

    for (j = 0; j < ds->nfeat; ++j) {
        res += (ds->var[j].type == CONTINUOUS) ? 1 : ds->var[j].levels;
    }

    return res;
}



void encode_instance(union data_point **X, uint32_t instance,
                     struct variable *var, uint16_t nvar,
                     double *enc)
{
    uint16_t j, l, idx;

    for (idx = j = 0; j < nvar; ++j) {
        if (var[j].type == CONTINUOUS) {
            enc[idx++] = X[j][instance].f;
        } else {
            for (l = 0; l < var[j].levels; ++l) {
                enc[idx++] = ((1 << l) == (1 << X[j][instance].c)) ? 1 : 0;
            }
        }
    }
}



void create_model_matrix(struct data_set *ds, uint32_t *subset, uint32_t nsamples,
                         double ***X_ptr, double ***Y_ptr,
                         double ***stats_ptr)
{
    uint32_t i;
    uint16_t j, J, C;
    double **X, **Y, **msd, delta;
    bool make_subset;

    J = encoding_size(ds);
    C = (ds->var[ds->nfeat].type == CONTINUOUS) ? 1 : ds->var[ds->nfeat].levels;

    X = MALLOC(nsamples, sizeof(double *));
    X[0] = MALLOC(nsamples * J, sizeof(double));

    Y = MALLOC(nsamples, sizeof(double *));
    Y[0] = MALLOC(nsamples * C, sizeof(double));

    make_subset = (subset == NULL);
    if (make_subset) {
        nsamples = ds->ninst;
        subset = MALLOC(nsamples, sizeof(int));
        for (i = 0; i < nsamples; ++i) subset[i] = i;
    }

    for (i = 0; i < nsamples; ++i) {
        X[i] = X[0] + i * J;
        encode_instance(ds->X, subset[i], ds->var, ds->nfeat, X[i]);

        Y[i] = Y[0] + i * C;
        if (ds->var[ds->nfeat].type == CONTINUOUS) {
            Y[i][0] = ds->y[subset[i]].f;
        } else {
            for (j = 0; j < C; ++j) Y[i][j] = 0;
            Y[i][ds->y[subset[i]].c] = 1;
        }
    }

    if (stats_ptr) {
        msd = MALLOC(J + 1, sizeof(double *));
        msd[0] = MALLOC((J + 1) * 2, sizeof(double));
        for (j = 0; j < (J + 1); ++j) {
            msd[j] = msd[0] + j * 2;
            msd[j][0] = msd[j][1] = 0;
        }

        for (i = 0; i < nsamples; ++i) {
            for (j = 0; j < J; ++j) {
                delta = X[i][j] - msd[j][0];
                msd[j][0] += delta / (i + 1);
                msd[j][1] += delta * (X[i][j] - msd[j][0]);
            }
            if (ds->var[ds->nfeat].type == CONTINUOUS) {
                delta = Y[i][0] - msd[J][0];
                msd[J][0] += delta / (i + 1);
                msd[J][1] += delta * (Y[i][0] - msd[J][0]);
            }
        }
        if (ds->var[ds->nfeat].type == CATEGORICAL) {
            msd[J][0] = NAN;
            msd[J][1] = NAN;
        }

        if (nsamples > 1) {
            for (j = 0; j < (J + 1); ++j) msd[j][1] = sqrt(msd[j][1] / (nsamples - 1));
        }

        for (i = 0; i < nsamples; ++i) {
            for (j = 0; j < J; ++j) if (msd[j][1] > 0) X[i][j] = (X[i][j] - msd[j][0]) / msd[j][1];
            if (ds->var[ds->nfeat].type == CONTINUOUS && (msd[J][1] > 0)) {
                Y[i][0] = (Y[i][0] - msd[J][0]) / msd[J][1];
            }
        }

        *stats_ptr = msd;
    }

    if (make_subset) free(subset);

    *X_ptr = X;
    *Y_ptr = Y;
}
