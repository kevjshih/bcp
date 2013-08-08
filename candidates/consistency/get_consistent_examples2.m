function [cached_scores reference_box] = get_training_examples(model, D, cached_scores, DISPLAY)
% Assumes cached_scores contains gt boxes (e.g. cached_gt)
% Does reverse process of get_consistent_examples, e.g., given part box, predict object bbox, and check overlap
part_todo = find(~[model.part.computed]);

model.part(part_todo).spat_const = [0 1 0.8 1 0 1];
model.part(part_todo).reference_box = [];
% Constants...
if(~exist('DISPLAY', 'var'))
    DISPLAY = 0;
end

NMS_IOU = 1;

cls = model.cls;

[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');

cached_pos = cached_scores(pos_inds);

% Collect positive hyps and their features ... %%%%%%%%%%%%%%%%%%%%%%%%%
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
pos_probs = sigmoid(pos_raw_scores, sig_param); %score2prec(roc, feat_all(:, end));
feat_all(:, end) = pos_probs;

%keyboard
%[idx box_centers] = cluster_bboxes(feat_all(:, 1:4), feat_all(:, end), 0.8);

% Get best object (hopefully the exemplar)
[dk best_obj] = max(pos_probs);

exemp_box = feat_all(best_obj, 1:4); % This is really dangerous, should be grounded to the actual initial exemplar!
ok = find(bbox_overlap_mex(feat_all(:, 1:4), exemp_box)>=0.5 & pos_probs>=0.3); % These thresholds are arbitrary

if(DISPLAY)
   BDglobals;
end

for pos_im = 1:length(pos_inds);
   im_ind = pos_inds(pos_im); % Get actual image index

   ok_reg_ind = inds_all(ok(inds_all(ok, 1)==pos_im), 3);
   ok_hyp_ind = inds_all(ok(inds_all(ok, 1)==pos_im), 2);

   %ok_reg_ind = inds_all(inds_all(:, 1)==pos_im, 3);
   %ok_hyp_ind = inds_all(inds_all(:, 1)==pos_im, 2);
      
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

reference_box = exemp_box;

% NOTHING HAPPENS DOWN HERE!
return;





[prob_sort sort_ind  ] = sort(pos_probs, 'descend');
box_sort = feat_all(sort_ind, 1:4);
inds_sort = inds_all(sort_ind, :);

boxes = zeros(length(prob_sort), 4);
box_inds = zeros(length(prob_sort), 1);
box_from = zeros(length(prob_sort), 1);

% Start by adding first box
boxes(1, :) = box_sort(1, :);
n_used = 1;
box_inds(1) = 1;
box_from(1) = 1;

ov_th = 0.8; % No idea if this is a good number
      draw_bbox([-0.5 -0.5 0.5 0.5]);
      hold on;
      draw_bbox(boxes(1:n_used, :));
for i = 2:length(prob_sort)
   i
   [best_ov best_ind] = max(bbox_overlap_mex(box_sort(i, :), boxes(1:n_used, :)), [], 2);

   if(best_ov>=ov_th) % It matches one of the existing boxes
      box_inds(i) = best_ind;
   else % No match, create new one
      fprintf('New box added!\n');
      n_used = n_used + 1;
      box_inds(i) = n_used;
      box_from(n_used) = i;
      boxes(n_used, :) = box_sort(i, :);

      draw_bbox(boxes(1:n_used, :));
   end

%   pause
end

% Now estimate the probability of each box:
counts = hist(box_inds, 1:max(box_inds));
acc = accumarray(box_inds, prob_sort);

box_probs = acc./counts(:);


keyboard
% Now cluster
[gmmp.mu gmmp.sigma gmmp.tau] = learn_gmm_w(feat_all(pos_probs>=0.2, :), pos_probs(pos_probs>=0.2), 3);
[gmmn.mu gmmn.sigma gmmn.tau] = learn_gmm_w(feat_all(pos_probs<0.2, :), 1-pos_probs(pos_probs<0.2), 3);

pred_pos = sum(apply_gmm(feat_all, gmmp.mu, gmmp.sigma, gmmp.tau), 2);
pred_neg = sum(apply_gmm(feat_all, gmmn.mu, gmmn.sigma, gmmn.tau), 2);


[inds centers dk dists] = kmeans([feat_all(:, 1:end-1) feat_all(:, end)], 2); % Cluster into two groups
[inds centers dk dists] = kmeans([feat_all(:, 1:end-1)], 2); % Cluster into two groups


start_mat = [mean(feat_all(feat_all(:,end)>0.8, :)); mean(feat_all(feat_all(:, end)<0.2, :))];
[inds centers dk dists] = kmeans([feat_all(:, 1:end-1) feat_all(:, end)], 2, 'Start', start_mat); % Cluster into two groups
[dk pos_clust] = max(centers(:, end)); % Cluster with largest score

if(DISPLAY)
   BDglobals;

   figure(1)
    ov = bbox_overlap_mex(box_sort(:, :), boxes(1, :));
   ok = find(ov>=0.80);
   [a r] = sort(prob_sort(ok), 'descend');

   for outer = 1:floor(length(a)/25)
   fprintf('working...')
   for hi = (outer-1)*25 + [1:25]
      i = ok(r(hi));
      ind = inds_sort(i, :);

      subplot(5,5, hi-(outer-1)*25);
      im = imread(fullfile(im_dir, Dpos(ind(1)).annotation.filename));
      imagesc(im);
      axis image; axis off;
      hold on;
      draw_bbox(cached_pos{ind(1)}.regions(ind(3), :));
      draw_bbox(pos_hyp{ind(1)}(ind(2)).bbox, 'r');
      title(sprintf('%f', a(hi)));
   end
   fprintf('Done\n')
   pause
   end
   % SHow originals
   figure(2)
   %ok = find(box_inds==1);
   [a r] = sort(prob_sort, 'descend');

   for hi = 1:25
      i = r(hi);
      ind = inds_sort(i, :);

      subplot(5,5, hi);
      im = imread(fullfile(im_dir, Dpos(ind(1)).annotation.filename));
      imagesc(im);
      axis image; axis off;
      hold on;
      draw_bbox(cached_pos{ind(1)}.regions(ind(3), :));
      draw_bbox(pos_hyp{ind(1)}(ind(2)).bbox, 'r');

   end
end

function [feats hyp_inds] = hyp_to_layout(pos_hyp, cached_pos, part_todo)

for i = 1:length(pos_hyp)
   if(~isempty(pos_hyp{i}))
      hyp = pos_hyp{i};

      ok_regions = find(cached_pos{i}.labels>=1);

      for h = 1:length(hyp)
         % This is the part box now
         gt_bbox = hyp(h).bbox(part_todo,:); %cached_pos{i}.regions(ok_regions(hyp(h).region), :);
         gt_dim = [gt_bbox(3:4) - gt_bbox(1:2)];
         gt_cent = 1/2*[gt_bbox(3:4) + gt_bbox(1:2)];

         scale = sqrt(sum(gt_dim.^2)); % scale defined as length of diagonal

         % Using GT box now
         bbox = cached_pos{i}.regions(ok_regions(hyp(h).region), :); %hyp(h).bbox(part_todo, :);

         %bbox_sc = (bbox - [gt_cent gt_cent])./([gt_dim gt_dim]);
         bbox_sc = (bbox - [gt_cent gt_cent])/scale;
            
         flipped = hyp(h).loc(4)==2;
         
         if(flipped) % This doesn't change
            bbox_sc = [-1 1 -1 1].*bbox_sc([3 2 1 4]); % Swap L/R coordinates and sign (since we're basically flipping around origin)
         end

         feats{i}(h, :) = [bbox_sc hyp(h).final_score];

         hyp_inds{i}(h, :) = [i, h, ok_regions(hyp(h).region)];
      end
   end
end

