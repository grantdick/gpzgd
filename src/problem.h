#ifndef PROBLEM_H
#define PROBLEM_H

#ifdef  __cplusplus
extern "C" {
#endif
    #include <stdbool.h>
    #include <stdint.h>

    #include "data_set.h"
    #include "rng.h"

    struct data_set *load_problem(const char *name, const char *problem_dir, double train_frac,
                                  uint32_t *n_leave_in_ptr, uint32_t **leave_in_ptr,
                                  uint32_t *n_hold_out_ptr, uint32_t **hold_out_ptr,
                                  rng_state state);

#ifdef  __cplusplus
}
#endif

#endif
