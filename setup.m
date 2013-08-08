% This script compiles all relevant mex files 

cd learning
mex svm_dual_hik_mex.cc
mex svm_dual_mex.cc -lmwblas
mex svm_weighted_dual_mex_test.cc
mex svm_weighted_dual_mex.cc -lmwblas
cd ../
cd external
!sh checkoutJavaBoost.sh

mex cummax_mex.cc
mex bbox_overlap_mex.cc
cd fast-additive-svms
make
cd ../../
cd candidates/quantombone-exemplarsvm-850fcb0/features
compile
cd ../../../
cd inference
mex features_w.cc
mex features.cc
mex get_best_part.cc
mex get_best_part_spat.cc
mex get_best_part_spat_new2.cc
mex get_best_part_spat_new.cc
mex IEresize.cc

cd ../
