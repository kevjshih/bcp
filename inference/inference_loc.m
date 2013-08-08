function [hyp feat_data] = inference_loc(input, model, region_inds, incr_feat)

BDglobals;
region_feat_dir = fullfile(dirs.feat_dir, 'region');
% For now, assume only 1 "part" is trained at a time
todo = find(~[model.part.computed]);

% Load the data ...
[dk bn dk] = fileparts(input); 
region_file = fullfile(region_feat_dir, [bn '_regfeat.mat']);

feats = load(region_file, 'region_hog');

reg_hog = cell(1, length(feats.region_hog)); 

if(~exist('region_inds','var'))
   region_inds = 1:length(feats.region_hog);
end

for j = reshape(find(region_inds), 1, []) %
   reg_hog{j} = feats.region_hog{j}(:);
end
reg_hog = cat(2, reg_hog{:});

if(~exist('incr_feat', 'var'))
   incr_feat = zeros(length(region_inds), numel(model.cached_weight));
end

% Apply model
scores = (model.part(todo).filter(:)'*reg_hog)' + incr_feat*model.cached_weight(:) + model.part(todo).bias;

ok_scores = find(scores >= model.thresh);

hyp = struct('score', [], 'final_score', [], 'region', [], 'reg_ind', [], 'bbox', [], 'loc', [], 'computed', [], 'feat_ind', []);
feat_data = {};
% Hyp ok
for i = 1:length(ok_scores)
   ind = ok_scores(i);
   hyp(i).score = incr_feat(ind, :);
   hyp(i).final_score = scores(ind);
   hyp(i).region = ind;
   hyp(i).reg_ind = region_inds(ind);
   hyp(i).feat_ind = i;
   hyp(i).bbox = [];
   hyp(i).loc = [];
   hyp(i).computed = 1;
%   hyp(i)
   feat_data{i} = reg_hog(:, ind);
end
