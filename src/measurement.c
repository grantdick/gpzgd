#include <stdint.h>

#include <math.h>

#include "data_set.h"
#include "measurement.h"

double compute_mse(union data_point *y, union data_point *yhat,
                    uint32_t *sample, uint32_t nsamples)
{
    double mse = 0;

    if (nsamples == 0) return NAN;

    for (uint32_t i = 0; i < nsamples; ++i) {
        double residual = yhat[sample[i]].f - y[sample[i]].f;
        mse += ((residual * residual) - mse) / (i + 1);
    }

    return mse;
}

double compute_rmse(union data_point *y, union data_point *yhat,
                    uint32_t *sample, uint32_t nsamples)
{
    return sqrt(compute_mse(y, yhat, sample, nsamples));
}

double compute_rrse(union data_point *y, union data_point *yhat,
                    uint32_t *sample, uint32_t nsamples)
{
    return sqrt(1 - compute_R2(y, yhat, sample, nsamples));
}

double compute_rsqr(union data_point *y, union data_point *yhat,
                    uint32_t *sample, uint32_t nsamples)
{
    double meanx = 0, meany = 0, C = 0, Sx = 0, Sy = 0;

    if (nsamples == 0) return NAN;

    for (uint32_t i = 0; i < nsamples; ++i) {
        double d1 = yhat[sample[i]].f - meanx;
        double d2 = y[sample[i]].f - meany;

        meanx += d1 / (i + 1);
        Sx += d1 * (yhat[sample[i]].f - meanx);

        meany += d2 / (i + 1);
        Sy += d2 * (y[sample[i]].f - meany);

        C += d1 * (y[sample[i]].f - meany);
    }

    return (C * C) / (Sx * Sy);
}

double compute_R2(union data_point *y, union data_point *yhat,
                  uint32_t *sample, uint32_t nsamples)
{
    double mse = 0, msd = 0, var = 0, residual, delta;

    if (nsamples == 0) return NAN;

    for (uint32_t i = 0; i < nsamples; ++i) {
        residual = y[sample[i]].f - yhat[sample[i]].f;
        mse += ((residual * residual) - mse) / (i + 1);

        delta = (y[sample[i]].f - msd);
        msd += delta / (i + 1);
        var += (y[sample[i]].f - msd) * delta;
    }

    var /= nsamples;

    return 1 - (mse / var);
}
