#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#ifdef  __cplusplus
extern "C" {
#endif

    #include <stdint.h>

    #include "data_set.h"

    double compute_mse(union data_point *y, union data_point *yhat,
                               uint32_t *sample, uint32_t nsamples);

    double compute_rmse(union data_point *y, union data_point *yhat,
                               uint32_t *sample, uint32_t nsamples);

    double compute_rrse(union data_point *y, union data_point *yhat,
                               uint32_t *sample, uint32_t nsamples);

    double compute_rsqr(union data_point *y, union data_point *yhat,
                               uint32_t *sample, uint32_t nsamples);

    double compute_R2(union data_point *y, union data_point *yhat,
                      uint32_t *sample, uint32_t nsamples);

#ifdef  __cplusplus
}
#endif

#endif
