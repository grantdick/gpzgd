#include <stdio.h>
#include <stdlib.h>

#include <stdint.h>

#include <math.h>

#include "cmd_args.h"
#include "data_set.h"
#include "gp.h"
#include "measurement.h"
#include "problem.h"
#include "readline.h"
#include "rng.h"

struct reporting_args {
    struct data_set *ds;

    uint32_t n_leave_in;
    uint32_t *leave_in;
    uint32_t *bootstrap;

    uint32_t n_hold_out;
    uint32_t *hold_out;

    union data_point *yhat;

    char * problem_name;
    uint16_t sample;
    uint16_t repeat;
};



static void print_predictions(union data_point **X, union data_point *t, union data_point *y,
                              uint32_t *hold_out, uint32_t n_hold_out, uint16_t p)
{
    uint32_t i, j;

    for (i = 0; i < n_hold_out; ++i) {
        fprintf(stdout, "%u %u", i + 1, hold_out[i]);
        for (j = 0; j < p; ++j) fprintf(stdout, " %g", X[j][hold_out[i]].f);
        fprintf(stdout, " %g %g\n", t[hold_out[i]].f, y[hold_out[i]].f);
    }
}


static void report_generation(struct gp **pop,
                              double *fitness __attribute__((unused)),
                              uint16_t best __attribute__((unused)),
                              uint16_t N,
                              struct gp *ret, uint16_t generation,
                              void *args_ptr)
{
    struct reporting_args args;
    double rmse_tr, rrse_tr;
    double rmse_ho, rrse_ho, rsqr_ho;
    double m, d;
    uint16_t i;

    args = *((struct reporting_args *)args_ptr);

    gp_predict(ret, args.ds->X, args.yhat, args.leave_in, args.n_leave_in);
    rmse_tr = compute_mse(args.ds->y, args.yhat, args.leave_in, args.n_leave_in);
    rrse_tr = compute_rrse(args.ds->y, args.yhat, args.leave_in, args.n_leave_in);

    gp_predict(ret, args.ds->X, args.yhat, args.hold_out, args.n_hold_out);
    rmse_ho = compute_mse(args.ds->y, args.yhat, args.hold_out, args.n_hold_out);
    rrse_ho = compute_rrse(args.ds->y, args.yhat, args.hold_out, args.n_hold_out);
    rsqr_ho = compute_rsqr(args.ds->y, args.yhat, args.hold_out, args.n_hold_out);
    if (!finite(rsqr_ho)) rsqr_ho = NAN;

    m = 0;
    for (i = 0; i < N; ++i) m += (gp_size(pop[i]) - m) / (i + 1);

    d = 0;
    for (i = 0; i < N; ++i) d += (gp_depth(pop[i]) - d) / (i + 1);

    fprintf(stdout, "%3u %g %g %g %g %g %u %g %u %g\n", generation, rmse_tr, rrse_tr, rmse_ho, rrse_ho, rsqr_ho, gp_size(ret), m, gp_depth(ret), d);
    fflush(stdout);
}

int main(int argc __attribute__((unused)), char **argv)
{
    struct main_parameters *main_params = default_main_parameters();
    struct gp_parameters *params = gp_default_parameters();

    struct data_set *ds;
    union data_point *yhat;
    uint32_t *leave_in, *hold_out, n_leave_in, n_hold_out;

    struct reporting_args rpt_args;

    struct gp *ind;

    /* first read in all the command line parameters and configuration files */
    load_config(argc - 1, argv + 1, main_params, params);

    seed_rng(main_params->rng_seed_file, main_params->rng_seed_offset, &(params->random_state));

    rpt_args.problem_name = main_params->problem_name;

    /* load problem */
    ds = load_problem(main_params->problem_name, main_params->problem_dir, main_params->train_frac, &n_leave_in, &leave_in, &n_hold_out, &hold_out, params->random_state);
    if (ds == NULL) {
        fprintf(stderr, "Unnown problem specified - please include the parameter problem=<problem_name> in config\n");
        exit(EXIT_FAILURE);
    }
    yhat = malloc(ds->ninst * sizeof(union data_point));

    rpt_args.ds = ds;
    rpt_args.n_leave_in = n_leave_in;
    rpt_args.leave_in = leave_in;
    rpt_args.n_hold_out = n_hold_out;
    rpt_args.hold_out = hold_out;
    rpt_args.yhat = yhat;

    params->rpt = main_params->verbose ? report_generation : NULL;
    params->rpt_args_ptr = &rpt_args;

    ind = gp_evolve(params, ds, leave_in, n_leave_in);

    if (main_params->print_hold_out) {
        fprintf(stdout, "###PREDICTIONS:\n");
        gp_predict(ind, ds->X, yhat, NULL, ds->ninst);
        print_predictions(ds->X, ds->y, yhat, hold_out, n_hold_out, ds->nfeat);
    }

    if (main_params->print_solution) {
        fprintf(stdout, "###SOLUTION: ");
        gp_print(stdout, ind);
        fprintf(stdout, "\n");
    }

    gp_free(ind);


    free(hold_out);
    free(leave_in);
    free(yhat);

    release_data_set(ds);

    gp_free_parameters(params);
    free_main_parameters(main_params);

    return EXIT_SUCCESS;
}
