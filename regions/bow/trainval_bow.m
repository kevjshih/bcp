function trainval_bow(cls)
%function train_bow_mc(features, labels)
% Binary classifiers only

if(1)
load_init_final; % Doing on train only for now....

C = 1;

%number of samples for uniform sampling of the functions
param.NSAMPLE = 100;
param.BINARY = 0;

[dk pos_inds] = LMquery(D, 'object.name', cls);
neg = 1:length(D);
neg(pos_inds) = [];

neg_sub = neg;
r = randperm(length(neg_sub));
neg = neg(r(1:min(end,200)));
Dsub = D([pos_inds(:); neg(:)]);
cached_sub = cached_scores([pos_inds(:); neg(:)]);
end


for iter = 1:4
   if(iter==1)
      [feat labels] = collect_bow_cls(Dsub, cached_sub, cls);

      iminds = {};
      for j = 1:length(feat)
         iminds{j} = repmat(j, size(feat{j},1), 1);
      end
      iminds = cat(1, iminds{:});
      cur_lab = cat(1, labels{:});
      features = cat(1, feat{:});
   else
      [feat_new labels_new] = collect_bow_cls(Dsub, cached_sub, cls, bowmodel{iter-1});
      iminds_new = {};
      for j = 1:length(feat_new)
         iminds_new{j} = repmat(j, size(feat_new{j},1), 1);
      end
      labels_new = cat(1, labels_new{:});
      feat_new = cat(1, feat_new{:});
      iminds_new = cat(1, iminds_new{:});

      [cur_lab features iminds] = update_cache_set(cur_lab, features, iminds, labels_new, feat_new, iminds_new, 0); % Don't update positive set, because there aren't any positives in this new data! 
   end 

   bowmodel{iter} = svmtrain_workingset(cur_lab(:), features, sprintf('-h 0 -t 5 -c %f -w1 %f -w-1 %f', C, 1, 1), param); 

   save(fullfile('data/results', cls, 'trainval_bow_model'), 'bowmodel');
end




return;
for i = length(model)+1:max(labels)
    fprintf('\n\n\nTraining model %d\n', i);
   cur_lab = 2*(labels==i)-1;

%   w_pos = 1/(2*sum(cur_lab==1));
%   w_neg = 1/(2*sum(cur_lab==-1));
  
   d{i} = svmpredict_approx(features, model{i}); % These will be used to train next layer
   save('data/bow/full_model.mat', 'model', 'd');
end

keyboard
mc_model = lintrain(labels, sparse(cat(2, d{:})), sprintf('-s 4 -c %f'));



