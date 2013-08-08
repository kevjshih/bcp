function [labels cached imind scores boxes regions] = prune_boost_data(labels_in, cached_scores, score_inds, ignore_boost_score, use_rank)

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
for i = 1:length(cached_scores)
   all_inds = [];
   if(isempty(cached_scores{i}.labels))
       continue;
   end
   
   ok_pos = find(cached_scores{i}.labels>0 & (~use_scores | any(~isinf(cached_scores{i}.part_scores(:, score_inds)), 2)));
   if(any(ok_pos))
      lab_ok = cached_scores{i}.labels(ok_pos);
      score_ok = cached_scores{i}.scores(ok_pos)*boost_score_coef + sum(cached_scores{i}.part_scores(ok_pos, score_inds),2) + ok_pos*(-use_rank); % Choose the highest scoring one
      best_ok_inds = ok_pos(get_best_hyp(lab_ok, score_ok));
      other_inds = [];%find(rand(size(ok_pos))<0.1); % Generate other postive examples
      all_inds = [all_inds(:); unique([best_ok_inds(:); other_inds(:)])];
   end

   if(INCLUDE_NEG)
      bad = find(cached_scores{i}.labels<=0 & (~use_scores | any(~isinf(cached_scores{i}.part_scores(:, score_inds)), 2)));
   else
      bad = find(cached_scores{i}.labels<0 & (~use_scores | any(~isinf(cached_scores{i}.part_scores(:, score_inds)), 2)));
   end

   if(any(bad) && (INCLUDE_NEG || ~any(cached_scores{i}.labels>0)))
      if(~isempty(bad))
         neg_scores = cached_scores{i}.scores(bad)*boost_score_coef + sum(cached_scores{i}.part_scores(bad, score_inds),2) + bad*(-use_rank); % Choose the highest scoring one
         % Rough NMS
         ROUGH_NMS = 0;
         if(ROUGH_NMS)
           [dk un] = unique(neg_scores);
            neg_inds = bad(un);
         else
           neg_inds = bad(nms_v4([cached_scores{i}.regions(bad,:), neg_scores], 0.5));
         end

         all_inds = [all_inds; neg_inds(:)];
      end
   end

   %cached{i} = [cached_scores{i}.part_scores(all_inds,:) all_inds(:)];
   cached{i} = [cached_scores{i}.part_scores(all_inds,:) cached_scores{i}.region_score(all_inds,:) all_inds(:)];
   labels{i} =  cached_scores{i}.labels(all_inds);
   scores{i} = cached_scores{i}.scores(all_inds);
   boxes{i} = cached_scores{i}.part_boxes(all_inds,:);
   regions{i} = cached_scores{i}.regions(all_inds,:);
   imind{i} = repmat(i, length(all_inds), 1);
end

labels = cat(1, labels{:});
labels(labels>0) = 1;
labels(labels<0) = -1;
cached = cat(1, cached{:});
scores = cat(1, scores{:});
boxes = cat(1, boxes{:});
regions = cat(1, regions{:});
imind = cat(1, imind{:});

function best_ind = get_best_hyp(true_label, scores)

   [ind dk un_label] =  unique(true_label);

   for i = 1:length(ind)
      this_ind = find(un_label==i);
      [best_score best_ind_t] = max(scores(this_ind));
      best_ind(i) = this_ind(best_ind_t);
   end

