#!/bin/sh

### Controls the number of threads per instance. Without this, each
### parallel instance will fire off its own set of threads,
### overwhelming the scheduler and causing havoc
export OMP_NUM_THREADS=1

mkdir -p runoutput

parallel --nice 19 --progress --eta \
	 ./dist/regression config/gecco2022/gpzgd.conf -p problem={1} -p train_frac={2} -p rng_seed_file=./problems/rngseeds/{1} -p rng_seed_offset={=2 '$_--'=} -p print_solution=Y '>' runoutput/{1}-{2}-gpzgd22 \
	 ::: F1 F2 F3 pagie1 abalone auto-mpg housing concrete dowchem energy ozone parkinsons tower winequality-red yacht \
	 ::: {1..50}

parallel -k \
	 awk -v problem=\'{1}\' -v method=\'gpzgd22\' \''!/###/ { print problem, method, $0 }'\' runoutput/{1}-{2}-gpzgd22 \
	 ::: F1 F2 F3 pagie1 abalone auto-mpg housing concrete dowchem energy ozone parkinsons tower winequality-red yacht \
	 ::: {1..50} \
	 > results-gecco2022

parallel -k \
	 awk -v problem=\'{1}\' -v method=\'gpzgd22\' \''BEGIN { OFS=";" } /###/ { gsub(/###SOLUTION: /, ""); print problem, method, $0 }'\' runoutput/{1}-{2}-gpzgd22 \
	 ::: F1 F2 F3 pagie1 abalone auto-mpg housing concrete dowchem energy ozone parkinsons tower winequality-red yacht \
	 ::: {1..50} \
	 > solutions-gecco2022

rm -rd runoutput
