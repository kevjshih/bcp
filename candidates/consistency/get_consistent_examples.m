function [cached_scores, reference_box, consistent_examples] = get_consistent_examples(model, D, cached_scores, DISPLAY)
% Assumes cached_scores contains gt boxes (e.g. cached_gt)

if(~isfield(model, 'min_ov'))
   min_ov = 0.75;
else
   min_ov = model.min_ov;
end

if(~isfield(model, 'min_prob'))
   min_prob = 0.3;
else
   min_prob = model.min_prob;
end



part_todo = find(~[model.part.computed]);

model.part(part_todo).spat_const = [0 1 0.8 1 0 1];
model.part(part_todo).reference_box = [];
% Constants...
if(~exist('DISPLAY', 'var'))
    DISPLAY = 0;
end

cls = model.cls;


% Get location of original exemplar part
% Since it may have come from outside of the current dataset, do some finagling to get the right data
if(~isempty(model.part(part_todo).name))
   [im part_bbox] = extract_exemplar_params(model.part(part_todo).name);
   D0 = pasc2D_id(im); % Get D structure

   bbox = LMobjectboundingbox(D0.annotation);
   [dk ok_bbox] = max(bbox_contained(part_bbox, bbox));

   exemplar_truncated = D0.annotation.object(ok_bbox).truncated;

   cached_tmp{1}.regions = bbox(ok_bbox, :);
   cached_tmp{1}.labels = 1;
   cached_tmp{1}.scores = 0;
   cached_tmp{1}.part_scores = [];zeros(1, model.num_parts);
   cached_tmp{1}.part_boxes = [];%: [500x0 double]

   [dk exemp_hyp] = collect_training_ex(model, D0, cached_tmp, 1);
else
   exemp_hyp{1} = [];
end

%warning('Removed default behavior of using exemplar box for consistency check\n');
if(~isempty(exemp_hyp{1}))
    exemp_feat = hyp_to_layout(exemp_hyp, cached_tmp, part_todo);
    exemp_box = exemp_feat{1}(1, 1:4);
    reference_box = exemp_box;
else
    exemplar_truncated = 1;  % Use clustering instead since it couldn't find a part for this exemplar
    reference_box = [];
end

% Collect positive hyps and their features ... %%%%%%%%%%%%%%%%%%%%%%%%%
[Dpos pos_inds] = LMquery(D, 'object.name', cls);
cached_pos = cached_scores(pos_inds);

[dk pos_hyp] = collect_training_ex(model, Dpos, cached_pos, 1);
[feats hyp_inds] = hyp_to_layout(pos_hyp, cached_pos, part_todo);

feat_all = cat(1, feats{:});
inds_all = cat(1, hyp_inds{:});


pos_raw_scores = feat_all(:, end);

% Collect negative hyps (for their scores) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
neg_inds = 1:length(D);
neg_inds(pos_inds) = [];
neg_inds = neg_inds(1:min(numel(Dpos),end));

% Don't care about examples past the last positive example
scores = sort(feat_all(:, end), 'descend');
model.thresh = scores(max(find(~isinf(scores))));

[dk neg_hyp] = collect_training_ex(model, D(neg_inds), cached_scores(neg_inds), -1);
neg_hyp = cat(1, neg_hyp{:});
neg_scores = unique([neg_hyp.final_score]);

% Evaluate results
roc = computeROC([feat_all(:, end); neg_scores(:)], [ones(size(feat_all,1),1); -ones(numel(neg_scores), 1)]);

% Learn and estimate probability of object
sig_param = learn_obj_prob(roc, 2);
% Unregularized logistic regression seems to work well!!
%sig_param = trainLogReg([feat_all(:, end); neg_scores(:)]', [ones(size(feat_all,1),1); -ones(numel(neg_scores), 1)]', 0.00000001);

pos_probs = sigmoid(pos_raw_scores, sig_param); %score2prec(roc, feat_all(:, end));
feat_all(:, end) = pos_probs;

%keyboard
%[idx box_centers] = cluster_bboxes(feat_all(:, 1:4), feat_all(:, end), 0.8);

% Get best object (hopefully the exemplar, fixed this to make sure it uses the exemplar)
%[dk best_obj] = max(pos_probs);

if(exemplar_truncated) % Testing this out for all examples
    [clust_inds clust_box] = kmeans(feat_all(:, 1:4), 10);
    
    % Find the cluster (or more generally coordinate frame) with the most
    % inliers
    ok_all = bsxfun(@and, bbox_overlap_mex(feat_all(:, 1:4), clust_box)>=min_ov, pos_probs>=min_prob);     
    
    [dk best_clust] = max(sum(ok_all, 1));
    ok0 = ok_all(:, best_clust);
    % Also make sure exemplar is included, if it's in the dataset
    if(exist('exemp_box', 'var'))
        ok0(bbox_overlap_mex(feat_all(:, 1:4), exemp_box)>=0.99) = 1;
    end
    
    box_to_use = clust_box(best_clust, :);
    exemp_box = box_to_use;
    ok = find(ok0);
else
    %exemp_box = feat_all(best_obj, 1:4); % This is really dangerous, should be grounded to the actual initial exemplar!
    box_to_use = exemp_box;
    ok = find(bbox_overlap_mex(feat_all(:, 1:4), exemp_box)>=min_ov & pos_probs>=min_prob); % These thresholds are arbitrary
end

overlap = bbox_overlap_mex(feat_all(:,1:4), box_to_use);

if(DISPLAY)
   BDglobals;
   
   %[dk sort_inds] = sort(feat_all(ok, end), 'descend');
   %todo = ok(sort_inds);
   
   [dk sort_inds] = sort(feat_all(:, end), 'descend');
   todo = sort_inds;
   
   for i = 1:length(todo)
      inds = inds_all(todo(i), :);
      ok_reg_ind = inds(3);
      ok_hyp_ind = inds(2);
      pos_im = inds(1);
      im_ind = pos_inds(inds(1));
      
      
      im = imread(fullfile(im_dir, D(im_ind).annotation.filename));
      clf;
      imagesc(im);
      hold on;
      if(ismember(todo(i), ok))
        draw_bbox(cached_scores{im_ind}.regions(ok_reg_ind, :));
      else
        draw_bbox(cached_scores{im_ind}.regions(ok_reg_ind, :), 'r');
      end
      
      hold on;
 %     draw_bbox(cached_scores{im_ind}.part_boxes);
      draw_bbox([pos_hyp{pos_im}(ok_hyp_ind).bbox(part_todo, :), bbox_overlap_mex(feat_all(todo(i), 1:4), box_to_use)],  '--');
      pause
      
   end
end

consistent_examples = [];

for pos_im = 1:length(pos_inds);
    im_ind = pos_inds(pos_im); % Get actual image index

    ok_reg_ind = inds_all(ok(inds_all(ok, 1)==pos_im), 3);
    ok_hyp_ind = inds_all(ok(inds_all(ok, 1)==pos_im), 2);
    ok_overlap = overlap(ok(inds_all(ok,1)==pos_im));
    ok_pos_prob = pos_probs(ok(inds_all(ok,1)==pos_im));

    for i = 1:size(ok_reg_ind,1)
        consistent_example.im_ind = im_ind;
        consistent_example.object_bbox = cached_scores{im_ind}.regions(ok_reg_ind(i),:);
        consistent_example.detection_bbox = pos_hyp{pos_im}(ok_hyp_ind(i)).bbox(part_todo,:);
        consistent_example.score = pos_hyp{pos_im}(ok_hyp_ind(i)).final_score;
        consistent_example.overlap = ok_overlap(i);
        consistent_example.pos_prob = ok_pos_prob(i);
        consistent_example.flipped = (pos_hyp{pos_im}(ok_hyp_ind(i)).loc(part_todo,4) == 2);

        consistent_examples = [consistent_examples; consistent_example];
    end

    % Display if flag is set
    if(DISPLAY & ~isempty(ok_reg_ind))
        im = imread(fullfile(im_dir, D(im_ind).annotation.filename));
        clf;
        imagesc(im);
        hold on;
        draw_bbox(cached_scores{im_ind}.regions(ok_reg_ind, :));
        hold on;
        %     draw_bbox(cached_scores{im_ind}.part_boxes);
        draw_bbox(cat(1, pos_hyp{pos_im}(ok_hyp_ind).bbox), '--');
        pause
    end
    cached_scores{im_ind} = prune_cached_scores(cached_scores{im_ind}, ok_reg_ind);
end

return;







function [feats hyp_inds] = hyp_to_layout(pos_hyp, cached_pos, part_todo)

for i = 1:length(pos_hyp)
   if(~isempty(pos_hyp{i}))
      hyp = pos_hyp{i};

      ok_regions = find(cached_pos{i}.labels>=1);

      for h = 1:length(hyp)
         gt_bbox = cached_pos{i}.regions(ok_regions(hyp(h).region), :);
         gt_dim = [gt_bbox(3:4) - gt_bbox(1:2)];
         gt_cent = 1/2*[gt_bbox(3:4) + gt_bbox(1:2)];

         scale = sqrt(sum(gt_dim.^2)); % scale defined as length of diagonal

         bbox = hyp(h).bbox(part_todo, :);

         bbox_sc = (bbox - [gt_cent gt_cent])./([gt_dim gt_dim]);
         %bbox_sc = (bbox - [gt_cent gt_cent])/scale;
            
         flipped = hyp(h).loc(part_todo, 4)==2;
         %flipped = hyp(h).loc(4)==2;
        
         if(flipped)
            bbox_sc = [-1 1 -1 1].*bbox_sc([3 2 1 4]); % Swap L/R coordinates and sign (since we're basically flipping around origin)
         end

         feats{i}(h, :) = [bbox_sc hyp(h).final_score];

         hyp_inds{i}(h, :) = [i, h, ok_regions(hyp(h).region)];
      end
   end
end

