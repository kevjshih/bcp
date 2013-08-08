function [cached_scores] = add_region_scores(model, D, cached_scores)

model.thresh = -inf; % Don't want to prune examples

dirs = [];
BDglobals;

%region_model = [];
%model_file = fullfile(WORKDIR, [model.cls '_region_model.mat']);
%load(model_file, 'region_model');

parfor ex = 1:length(D)
   fprintf('%d/%d\n', ex, length(D));
   [reg_scores{ex}] = update_example(D(ex).annotation, dirs, model.region_model); 
end


for ex = 1:length(D)
   if(~isempty(cached_scores{ex}.regions))
      cached_scores{ex}.region_score(:, 1:3) = reg_scores{ex};
   end
end

function reg_scores = update_example(ann, dirs, region_model)

   region_feat_dir = fullfile(dirs.feat_dir, 'region');
   [dk bn dk] = fileparts(ann.filename); 
   region_file = fullfile(region_feat_dir, [bn '_regfeat.mat']);

   if(~exist(region_file, 'file')) 
      if(extract_region_features(ann, dirs)==0)
         reg_scores = [];
         return;
      end
   end
      
   
   feats = load(region_file, 'c_hist', 't_hist', 'region_hog');
   c_hist = cat(2, feats.c_hist{:});
   t_hist = cat(2, feats.t_hist{:});
  
   reg_hog = cell(1, length(feats.region_hog)); 
   for j = 1:length(feats.region_hog)
      reg_hog{j} = feats.region_hog{j}(:);
   end
   reg_hog = cat(2, reg_hog{:});

   reg_scores = [region_model{1}*c_hist; region_model{2}*t_hist; region_model{3}*reg_hog]';
