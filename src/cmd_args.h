#ifndef CMD_ARGS_H
#define CMD_ARGS_H

#ifdef  __cplusplus
extern "C" {
#endif

    #include <stdbool.h>
    #include <stdint.h>

    #include "gp.h"

    struct main_parameters {
        char *   problem_dir;
        char *   problem_name;
        double   train_frac;

        char *   rng_seed_file;
        uint32_t rng_seed_offset;

        bool verbose;
        bool print_hold_out;
        bool print_solution;
    };

    struct main_parameters *default_main_parameters();
    void free_main_parameters(struct main_parameters *main_params);

    void load_config(int argc, char **argv, struct main_parameters *main_params, struct gp_parameters *params);

#ifdef  __cplusplus
}
#endif

#endif
