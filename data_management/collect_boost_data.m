function [labels cached_scores] = mine_examples(model, D, cached_scores)
if(~isfield(model, 'Ntodo'))
    model.Ntodo = 1;
end

model.thresh = -inf; % Don't want to prune examples

dirs = [];
BDglobals;

new_part_scores = cell(size(cached_scores));
new_part_boxes = cell(size(cached_scores));

%parfor_progress(length(D));
if(isfield(model, 'do_transform') && model.do_transform==1)
   new_part_trans = cell(size(cached_scores));

   
   parfor ex = 1:length(D)
       if(rand<50/length(D))
           fprintf('.\n');
       end
       
      [labels{ex} new_part_scores{ex} new_part_boxes{ex} new_part_trans{ex}] = mine_example(model, D(ex).annotation, cached_scores{ex}, dirs); 
   end

   for ex = 1:length(D)
      cached_scores{ex}.part_scores = new_part_scores{ex};
      cached_scores{ex}.part_boxes = new_part_boxes{ex};
      cached_scores{ex}.part_trans = new_part_trans{ex};
   end
else
   parfor ex = 1:length(D)
%    parfor_progress;
       if(rand<50/length(D))
           fprintf('.\n');
       end
      [labels{ex} new_part_scores{ex} new_part_boxes{ex}] = mine_example(model, D(ex).annotation, cached_scores{ex}, dirs); 
   end

   for ex = 1:length(D)
      cached_scores{ex}.part_scores = new_part_scores{ex};
      cached_scores{ex}.part_boxes = new_part_boxes{ex};
   end
end
%parfor_progress(0);

function [labels new_part_scores new_part_boxes new_part_trans] = mine_example(model, annotation, cached_scores, dirs)
   % To define: region_dir, feat_dir, im_dir
   cls = model.cls;

   [dk bn dk] = fileparts(annotation.filename); 
   feat_file = fullfile(dirs.feat_dir, [bn '.mat']);
   label_dir = fullfile(dirs.label_dir, cls);
   label_file = fullfile(label_dir, [bn '.mat']);
   region_file = fullfile(dirs.region_dir, [bn '_proposals.mat']);


   regions = cached_scores.regions;
   labels = cached_scores.labels;

   if(isfield(model, 'incremental_feat') && model.incremental_feat==1)
      scores = cached_scores.incremental_feat;
   else
      scores = cached_scores.scores;
   end
%   part_scores = cached_scores.part_scores;

   new_part_boxes = [];
   new_part_trans = [];
   if(numel(regions)==0)
      feats = [];
      hyp = [];
      labels = [];
      new_part_scores = [];
      new_part_boxes = [];
      new_part_trans = [];
      return;
   end


   % Find the hypothesis!
   if(0&&exist(feat_file, 'file')) % Reading the features is too slow
      load(feat_file, 'feat_data');
      im = feat_data;
      hyp = inference(im, model, regions, scores);
   else
      im_file = fullfile(dirs.im_dir, annotation.filename);

      if(isfield(model, 'bound_feat') && model.bound_feat==1)
         im = get_bound_feat(model, annotation.filename, dirs);
      else 
         im = imread(im_file);
      end
      [hyp] = inference(im, model, regions, scores);
      %save(feat_file, 'feat_data', '-v6');
   end
   
   % Compute features for each region
   %feats = hyp_to_feat(model, hyp, feat_data);
   if(~isfield(model, 'do_boxes'))
       DO_BOXES = 1;
   else
      DO_BOXES = model.do_boxes;
   end
    Ntodo = model.Ntodo;

   
   if(~isempty(hyp))
      computed = [model.part.computed]==1;

      if(isfield(model, 'score_feat') && model.score_feat==1)
         new_part_scores = repmat(cat(1,hyp.cached_score)*model.cached_weight(:), 1, model.num_parts); % Include 
      else
         new_part_scores = zeros(numel(hyp), model.num_parts, Ntodo);
      end

      if(DO_BOXES)
         new_part_boxes = zeros(numel(hyp), 4*model.num_parts, Ntodo, 'int16');
      end

      % Copy over cached scores
      if(sum(computed)>0 && size(cached_scores.part_scores,2)>=sum(computed))
         new_part_scores(:, computed, :) = cached_scores.part_scores(:, computed, :);
      end
      
      if(DO_BOXES && any(computed)) % In case 
         new_part_boxes(:, kron(computed, ones(1, 4))==1, :) = cached_scores.part_boxes(:,kron(computed, ones(1, 4))==1, :);
      end

      feats = cat(3, hyp.score);
      % Add new part scores
      new_part_scores(:, ~computed, :) = permute(feats(~computed,:, :), [3 1 2]) + new_part_scores(:, ~computed, :);

      new_boxes = zeros(numel(hyp), sum(~computed)*4, Ntodo);
      for b = 1:length(hyp)
         new_boxes(b,:, :) = reshape(permute(hyp(b).bbox(~computed,:, :), [2 1 3]), 1, [], Ntodo);
      end

      if(DO_BOXES)
         new_part_boxes(:, kron(~computed, ones(1,4))==1, :) = int16(new_boxes);

         if(isfield(model, 'do_transform') && model.do_transform==1)
            if(isfield(model, 'shift') && numel(model.shift)>1)
               K = 4; % Flip, xsh, ysh, rot placeholder
            else
               K = 1; % Only one transformation right now
            end

            new_part_trans = zeros(numel(hyp), K*model.num_parts, Ntodo,  'int16');
            if(any(computed) && isfield(cached_scores, 'part_trans'))
               new_part_trans(:, kron(computed, ones(1, K))==1, :) = cached_scores.part_trans(:,kron(computed, ones(1, K))==1, :);
            end
         
            new_trans = zeros(numel(hyp), sum(~computed)*K, Ntodo);
            for b = 1:length(hyp)
               new_trans(b,:, :) = permute(hyp(b).loc(~computed, 3 + [1:K], :), [2 1 3]); %, , 1, [], Ntodo);
            end
            new_part_trans(:, kron(~computed, ones(1,K))==1, :) = int16(new_trans);
         end
      end
   else
      feats = [];
      new_part_scores = [];
      new_part_boxes = [];
      new_part_trans = [];
   end
   

