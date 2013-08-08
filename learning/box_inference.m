function cached_scores = box_inference(cached_scores, model) %box_offsets, part_weights)

box_offsets = model.box_model.box_offsets;
part_weights = model.box_model.part_weights;
all_part_weights = cat(2, part_weights{:});
calib_params = {model.part.calib};

box_pred_ind = find(strcmp(model.region_model, 'pred_box_overlap'));

DO_REG = 1;

for i = 1:length(cached_scores)
   if(isempty(cached_scores{i}.regions)), continue; end

   all_pred = {};
   all_weights = {};
   all_flipped = {};
   
   reg = cached_scores{i}.regions;
   for j = 1:model.num_parts + DO_REG
       if(numel(box_offsets)<j || isempty(box_offsets{j}))
           all_pred{i,j} = zeros(size(reg,1), 4);
           all_weights{i,j} = zeros(size(reg,1), 1);
           continue;
       end
       
      if(j==model.num_parts+1) % Use the region!
         part_box = reg; %cached_scores{i}.regions;
         flipped = false(size(reg,1), 1);
         probs = ones(size(reg,1), 1);
      else
         part_box = double(cached_scores{i}.part_boxes(:, 4*(j-1) + [1:4]));
         flipped = double(cached_scores{i}.part_trans(:, j))==2;
         probs = sigmoid(cached_scores{i}.part_scores(:, j), calib_params{j});

      end

      %diag = sqrt(sum((part_box(:, [3 4]) - part_box(:, [1 2])).^2, 2));
      diag = repmat(part_box(:, [3 4]) - part_box(:, [1 2]), [1 2]);
      cent = 1/2*(part_box(:, [3 4]) + part_box(:, [1 2]));

      pred_box0 = repmat(box_offsets{j}, size(cent,1), 1);
      pred_box0(flipped, :) = bsxfun(@times, pred_box0(flipped, [3 2 1 4]), [-1 1 -1 1]);
      %pred_box = bsxfun(@times, pred_box0, diag) + [cent cent];
      pred_box = pred_box0.*diag + [cent cent];

      all_pred{j} = pred_box;
      all_weights{j} = probs; 
      all_flipped{j} = flipped;
   end
   % Now average boxes
   new_box = zeros(size(cached_scores{i}.regions,1), 4);

   boxes = cat(3,all_pred{:});
   weights = cat(2, all_weights{:});
   weights(sum(weights, 2)<1e-5, :) = 1;
   flip = cat(2, all_flipped{:});
   
   if(isfield(model.box_model, 'do_flip') && model.box_model.do_flip==1)
      for j = 1:size(boxes,1)
         flipped_weights = all_part_weights;
         flipped_weights(flip(j, :)==1, [1 3]) = flipped_weights(flip(j, :)==1, [3 1]);

         B = permute(boxes(j, :, :), [3 2 1]);
         probs = weights(j, :)';

         Bp = bsxfun(@times, B, probs);
         num = sum(flipped_weights.*Bp, 1)';
         Z = flipped_weights'*probs;
         new_box(j, :) = num./Z;
      end
   else
      for p = 1:4 
         X = [permute(boxes(:, p, :), [1 3 2]), weights];
         new_box(:, p) = box_fit(part_weights{p}, X);
      end
   end
   %for j = 1:size(cached_scores{i}.regions, 1)
   %   cached_scores{i}.region_score(j, box_pred_ind) = bbox_overlap_mex(cached_scores{i}.regions(j, :), new_box{i}(j, :));
   %end

   cached_scores{i}.regions = new_box; 
end
