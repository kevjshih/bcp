if(~exist('Y','var'))
   load ../data/bow/tmp_bow_features_train.mat
   Y = 2*(labels(1:end/2)>0) - 1;
   X = features(:, 1:end/2);
   Yte = 2*(labels(end/2+1:end)>0) - 1;
   Xte = features(:, end/2+1:end);

   clear features;
end

Nbin = 100;
C = 1;


[model alphas] = svm_dual_hik(Y, X, C, Nbin);


return;
addpath(genpath('~/prog/tools/fast-additive-svms/'))
model0 = svmtrain(Y, X', sprintf('-s 0 -t 5 -c %f', C));
