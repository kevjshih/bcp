function [cached_scores] = add_region_scores(model, D, cached_scores)

model.thresh = -inf; % Don't want to prune examples

dirs = [];
BDglobals;

%region_model = [];
%model_file = fullfile(WORKDIR, [model.cls '_region_model.mat']);
%load(model_file, 'region_model');

for ex = 1:length(D)
   fprintf('%d/%d\n', ex, length(D));
   [reg_scores{ex}] = update_example(D(ex).annotation, dirs, model.box_local_model); 
end

inds = length(model.region_model) + 1;

for ex = 1:length(D)
   if(~isempty(cached_scores{ex}.regions))
      cached_scores{ex}.region_score(:, inds) = reg_scores{ex};
   end
end

function reg_scores = update_example(ann, dirs, region_model)

   region_feat_dir = fullfile(dirs.feat_dir, 'box');
   [dk bn dk] = fileparts(ann.filename); 
   region_file = fullfile(region_feat_dir, [bn '_boxfeat.mat']);

   if(~exist(region_file, 'file'))
      reg_scores = [];
      return;
   end
      
   
   feats = load(region_file, 'box_feat');
   box_hog = cell(1, length(feats.box_feat)); 

   if(isempty(box_hog))
      reg_scores = [];
      return;
   end

   for j = 1:length(box_hog)
      box_hog{j} = feats.box_feat{j}(:);
   end
   box_hog = cat(2, box_hog{:});

   reg_scores = [region_model(:)'*box_hog]';
