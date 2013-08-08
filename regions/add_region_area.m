function [cached_scores] = add_region_scores(model, D, cached_scores)

model.thresh = -inf; % Don't want to prune examples

dirs = [];
BDglobals;

%region_model = [];
%model_file = fullfile(WORKDIR, [model.cls '_region_model.mat']);
%load(model_file, 'region_model');

for ex = 1:length(D)
   fprintf('%d/%d\n', ex, length(D));
   [reg_scores{ex}] = update_example(D(ex).annotation, cached_scores{ex}, dirs); 
end

ind = [];
for i = 1:length(model.region_model)
   if(ischar(model.region_model{i}) && strcmp(model.region_model{i}, 'Area model'))
      ind = [ind i];
   end
end

for ex = 1:length(D)
   if(~isempty(cached_scores{ex}.regions))
       cached_scores{ex}.region_score(:, ind) = reg_scores{ex};
   end
end

function reg_scores = update_example(ann, cached_scores, dirs)

   st = imfinfo(fullfile(dirs.im_dir, ann.filename));      
   
   norm = sqrt(st.Width.^2 + st.Height.^2);

   reg = cached_scores.regions;
   % normalized length of diagonal, aspect ratio
   if(isempty(reg))
       reg_scores = [];
   else
      reg_scores = [sqrt(sum((reg(:, [3 4]) - reg(:, [1 2])).^2,2))/norm,  log(reg(:, 3) - reg(:, 1)) - log(reg(:, 4) - reg(:, 2))];
   end