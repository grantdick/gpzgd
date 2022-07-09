#ifndef GP_H
#define GP_H

#ifdef  __cplusplus
extern "C" {
#endif

    #include <stdio.h>

    #include <stdbool.h>
    #include <stdint.h>

    #include "data_set.h"
    #include "rng.h"

    enum gp_operator { ADD, SUB, MUL, DIV, AQT, PDIV, SIN, COS, EXP, LOG, INV, PEXP, PLOG, PINV, ERC, VAR };

    struct gp;

    struct gp_parameters
    {
        rng_state random_state;

        double validation_prop;

        uint16_t pop_size;
        uint16_t generations;

        double elitism_rate;
        double tournament_size;
        bool   lexicographic_tournament;

        double crossover_rate;
        double sub_mutation_rate;
        double point_mutation_rate;

        uint8_t min_tree_init;
        uint8_t max_tree_init;
        int8_t max_tree_depth;
        int32_t max_tree_nodes;

        uint8_t num_functions;
        uint8_t num_terminals;
        enum gp_operator *ops;

        bool standardise;

        bool coef_op;
        double learning_rate;
        uint16_t learning_epochs;

        double mutation_sigma;

        void (*rpt)(struct gp **, double *, uint16_t, uint16_t, struct gp *, uint16_t, void *);
        void *rpt_args_ptr;

        uint32_t timeout;
    };

    struct gp_parameters *gp_default_parameters();

    void gp_free_parameters(struct gp_parameters *params);

    void gp_init_function_set(struct gp_parameters *parms, const char *op_list);

    struct gp *gp_evolve(struct gp_parameters *params, struct data_set *ds, uint32_t *subset, uint32_t nsamples);

    void gp_free(struct gp *ind);

    void gp_predict(struct gp *ind,
                    union data_point **X, union data_point *yhat,
                    uint32_t *subset, uint32_t nsamples);

    void gp_print(FILE *out, struct gp *ind);

    uint16_t gp_size(struct gp *ind);
    uint8_t gp_depth(struct gp *ind);

#ifdef  __cplusplus
}
#endif

#endif
