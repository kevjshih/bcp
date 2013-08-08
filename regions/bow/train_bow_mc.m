function train_bow_mc(features, labels)
% Binary classifiers only

C = 1;

    

%number of samples for uniform sampling of the functions
param.NSAMPLE = 100;
param.BINARY = 0;

model = {};

if(exist('data/bow/full_model.mat', 'file'))
    load('data/bow/full_model.mat', 'model', 'd');
end


for i = length(model)+1:max(labels)
    fprintf('\n\n\nTraining model %d\n', i);
   cur_lab = 2*(labels==i)-1;

%   w_pos = 1/(2*sum(cur_lab==1));
%   w_neg = 1/(2*sum(cur_lab==-1));
  
   model{i} = svmtrain_workingset(cur_lab(:), features, sprintf('-h 0 -t 5 -c %f -w1 %f -w-1 %f', C, 1, 1), param); 
   d{i} = svmpredict_approx(features, model{i}); % These will be used to train next layer
   save('data/bow/full_model.mat', 'model', 'd');
end

keyboard
mc_model = lintrain(labels, sparse(cat(2, d{:})), sprintf('-s 4 -c %f'));



