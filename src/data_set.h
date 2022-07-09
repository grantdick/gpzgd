#ifndef DATA_SET_H
#define DATA_SET_H

#ifdef  __cplusplus
extern "C" {
#endif

    #include <stdbool.h>
    #include <stdint.h>

    enum feature_type { CONTINUOUS, CATEGORICAL };

    union data_point {
        double  f; /* continuous/ordinal value, set to NaN for missing */
        int16_t c; /* categorical value id, -1 to indicate missing */
    };

    struct variable {
        char *name;             /* names of the features, if NULL,
                                 * then x1, x2, ..., xN, and y will be
                                 * used in place of the names */

        enum feature_type type; /* categorical or continuous */

        uint16_t levels;        /* the number of levels in the feature
                                 * (or zero, if continuous) */

        char **labels;          /* the labels of the categories, or
                                 * NULL if the variable is
                                 * continuous */
    };

    struct data_set {
        uint32_t ninst;         /* number of instances */
        uint16_t nfeat;         /* number of features */

        struct variable *var;   /* column information (inc. output) */

        union data_point **X;   /* input variables, in column-major order */
        union data_point  *y;   /* output variable */
    };

    struct variable *copy_variable_info(struct variable *var, uint16_t nvar);

    struct data_set *load_data(const char *source);

    struct data_set *empty_data_set();

    void release_data_set(struct data_set *ds);

    void release_variable_info(struct variable *var, uint16_t nvar);

    /* splits the data set into two subsets - a keep-in set and
     * hold-out set. Puts the results of splitting into the two out
     * parameters keepin and holdout - these are deep copies of the
     * data, so be careful with very large data sets (consider using
     * subset indices instead) */
    void split_data(struct data_set *ds, uint8_t *partition,
                    struct data_set **keepin_ptr, struct data_set **holdout_ptr);

    /* performs an in-place sort of the idx array, according to the
     * corresponding values in x. the values in the idx array are the
     * corresponding entries in x, so the size of x must be at least
     * equal to the largest entry in idx */
    void sort_index(union data_point *x, uint32_t *idx, uint32_t nsamples, enum feature_type type);

    bool missing_value(union data_point x, enum feature_type type);

    int32_t compare_values(union data_point x, union data_point y,
                           enum feature_type type, bool compl);

    union data_point unknown_value(enum feature_type type);

    void create_model_matrix(struct data_set *ds, uint32_t *subset, uint32_t nsamples,
                             double ***X_ptr, double ***Y_ptr,
                             double ***stats_ptr);

    uint16_t encoding_size(struct data_set *ds);

    void encode_instance(union data_point **X, uint32_t instance,
                         struct variable *var, uint16_t nvar,
                         double *enc);

#ifdef  __cplusplus
}
#endif

#endif
