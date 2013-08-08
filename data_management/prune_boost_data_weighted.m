function [labels cached scores boxes regions] = prune_boost_data(labels_in, cached_scores, score_inds, ignore_boost_score, use_rank)
% Only use 1 part score
error('Don''t use this function!');
if(~exist('score_inds','var'))
   score_inds = [];
end

if(~exist('ignore_boost_score', 'var'))
   ignore_boost_score = 0;
end

if(~exist('use_rank', 'var'))
   use_rank = 0;
end

boost_score_coef = double(~ignore_boost_score);
use_scores = double(~isempty(score_inds));

INCLUDE_NEG = 1;
for i = 1:length(labels_in)
   all_inds = [];
   if(isempty(labels_in{i}))
       continue;
   end
  
   % Positive example:
   % needs positive label
   % if using scores, at least one part needs to be non-inf
   ok_pos = find(labels_in{i}>0 & (~use_scores | any(~isinf(cached_scores{i}.part_scores(:, 1:score_inds)), 2)));
   if(any(ok_pos))
      lab_ok = labels_in{i}(ok_pos);

      prev_scores = cached_scores{i}.scores(ok_pos)*boost_score_coef;
      new_scores = sum(cached_scores{i}.part_scores(ok_pos, score_inds),2); % Choose the highest scoring one
      best_ok_inds = ok_pos(get_best_pos_hyp(lab_ok, prev_scores, new_scores, ok_pos));

      all_inds = [all_inds(:); best_ok_inds(:)];
   end


   bad = find(labels_in{i}<=0 & (~use_scores | any(~isinf(cached_scores{i}.part_scores(:, 1:score_inds)), 2)));
   if(any(bad) && (INCLUDE_NEG || ~any(labels_in{i}>0)))
      if(~isempty(bad))
         prev_scores = cached_scores{i}.scores(bad)*boost_score_coef;
         new_scores = sum(cached_scores{i}.part_scores(bad, score_inds),2);
         
         neg_inds = bad(get_best_neg_hyp(cached_scores{i}.regions(bad,:), prev_scores, new_scores, bad));
%           neg_inds = bad(nms_v4([cached_scores{i}.regions(bad,:), prev_scores], 0.5));

         all_inds = [all_inds; neg_inds(:)];
      end
   end

   cached{i} = [cached_scores{i}.part_scores(all_inds,:) all_inds(:)];
   labels{i} = labels_in{i}(all_inds);
   scores{i} = cached_scores{i}.scores(all_inds);
   boxes{i} = cached_scores{i}.part_boxes(all_inds,:);
   regions{i} = cached_scores{i}.regions(all_inds,:);
end

labels = cat(1, labels{:});
labels(labels>0) = 1;
labels(labels<=0) = -1;
cached = cat(1, cached{:});
scores = cat(1, scores{:});
boxes = cat(1, boxes{:});
regions = cat(1, regions{:});

function best_ind = get_best_neg_hyp(boxes, prev_scores, new_scores, rank)


% First suppress ones that have a new part
ok_new_score = ~isinf(new_scores);

already_selected = [];
already_selected_boxes = zeros(0,5);
if(any(ok_new_score))
   ok_ind = find(ok_new_score);
   ok_score = 1./(1+exp(-prev_scores(ok_ind))) .* new_scores(ok_ind);
   selected = nms_v4([boxes(ok_ind,:) ok_score], 0.5);

   already_selected = ok_ind(selected);
   already_selected_boxes = [boxes(selected,:), repmat(max(prev_scores)+100, numel(already_selected), 1)]; % Make sure these are selected in the next round
end

% 
ok_old_score = find(~ok_new_score);
to_select = [already_selected; ok_old_score];

if(~isempty(ok_old_score))
   final_selected = nms_v4([already_selected_boxes; [boxes(ok_old_score,:), prev_scores(ok_old_score)]], 0.5);
else
   final_selected = 1:length(already_selected);
end
best_ind = to_select(final_selected);


function best_ind = get_best_pos_hyp(true_label, prev_scores, new_scores, rank)

   [ind dk un_label] =  unique(true_label);

   for i = 1:length(ind)
      this_ind = find(un_label==i);
      if(any(~isinf(new_scores(this_ind)))) % Use the newest part weighted by previous score
         score_to_check = 1./(1+exp(prev_scores(this_ind))) .* new_scores(this_ind);
      else % Just use previous weights to select example
         score_to_check = prev_scores(this_ind);
      end

      [best_score best_ind_t] = max(score_to_check);
      % Use rank to break ties, this should be automatic since examples are sorted by rank...
      if(~isempty(rank))
         rank_for_this = rank(this_ind);
         ties = find(abs(score_to_check-best_score)<1e-6);
         [best_rank best_among_ties] = min(rank_for_this(ties));      

         best_ind_t = ties(best_among_ties);
      end

      best_ind(i) = this_ind(best_ind_t);
   end

