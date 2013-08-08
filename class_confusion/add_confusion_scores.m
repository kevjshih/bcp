function cached_scores = add_confusion_scores(model, cached_scores, D, cc_model, doloo)

model.thresh = -inf;

dirs = [];
BDglobals;

new_scores = cell(numel(D), 1);

if(~exist('doloo','var'))
   doloo = 0;
end

parfor i = 1:length(D)
   fprintf('%d\n', i);
   if(~isempty(cached_scores{i}.scores))
      new_scores{i} = helper(D(i).annotation, cached_scores{i}, model, cc_model, dirs, doloo*i);
   end
end

for i = 1:length(D) % Copy results back over
   if(~isempty(cached_scores{i}.scores))
      cached_scores{i}.part_scores = [cached_scores{i}.part_scores new_scores{i}];
   end
end

function new_scores = helper(ann, cached_scores, model, cc_model, dirs, doloo)
% this function treats one example/image
% doloo chooses the cc svm which left doloo-th example out during training
% 

% Collect the indices for part inference
ind = zeros(length(cc_model), 1);
for i = 1:length(cc_model)
   ind(i) = cc_model{i}(1).part_ind; 
end

[model.part.computed] = deal(1);
[model.part(ind).computed] = deal(0); % Only use features for relevant parts

im = imread(fullfile(dirs.im_dir, ann.filename));
[hyp feat_data] = inference(im, model, cached_scores.regions);

for i = 1:length(cc_model)
   [model.part.computed] = deal(1);
   model.part(ind(i)).computed = 0; % Only use features for current part
   
   color_pyramid_max_lvl = 4;
   feat = hyp_to_auxfeat(model, hyp, feat_data, im, ind(i), color_pyramid_max_lvl);
   %feat = hyp_to_feat(model, hyp, feat_data);

   models = {};
   for j = 1:length(cc_model{i})
      if(doloo==0 || length(cc_model{i}(j).w_loo)<doloo || isempty(cc_model{i}(j).w_loo{doloo}))
         models{j} = cc_model{i}(j).w_noloo;
      else
         models{j} = cc_model{i}(j).w_loo{doloo};
      end
   end

   models = cat(1, models{:})';
   %size_of_feat = size(feat);
   %size_of_models = size(models);
   %fprintf('feat size %d %d\n',size_of_feat(1),size_of_feat(2));
   %fprintf('models size %d %d\n',size_of_models(1),size_of_models(2));
   
   scores{i} = feat'*models(1:end-1,:);

%   scores{i}([hyp.final_score]<cc_model{i}(1).thresh, :) = -inf;
end

new_scores = cat(2, scores{:});
