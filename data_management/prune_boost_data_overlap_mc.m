function [labels cached imind scores boxes regions] = prune_boost_data_ovelap_mc(D, cached_scores, cls)
%function [labels cached imind scores boxes regions] = prune_boost_data_ovelap_mc(D, cached_scores, cls)

INCLUDE_NEG = 0;
for i = 1:length(cached_scores)
   all_inds = [];
   cur_labels = [];
   if(isempty(cached_scores{i}.labels))
       continue;
   end
   
   annotation = D(i).annotation;
   if(any(cached_scores{i}.labels(:)>0)) % positive image
      for cl = 1:length(cls)
         boxes = LMobjectboundingbox(annotation, cls{cl});
         if(isempty(boxes))
            continue;
         end
         [overlaps best_ind] = max(bbox_overlap_mex(boxes, cached_scores{i}.regions), [], 2);
   
         pos_inds = best_ind(overlaps>0.5);
         all_inds = [all_inds; pos_inds(:)];
         labels_t = -ones(length(pos_inds), length(cls));
         labels_t(:, cl) = 1;
         cur_labels = [cur_labels; labels_t];
      end 
   end

   if(INCLUDE_NEG)
      bad = find(all(cached_scores{i}.labels<=0,2));
   else
      bad = find(all(cached_scores{i}.labels<0, 2));
   end 

   if(any(bad) && (1 || ~any(cached_scores{i}.labels>0)))
   %if(any(bad) && (INCLUDE_NEG || ~any(cached_scores{i}.labels>0)))
%   if(any(bad) && any(cached_scores{i}.labels>0)))
      if(~isempty(bad))
         neg_scores = bad; % Break ties with rank

         neg_inds = bad(nms_iou([cached_scores{i}.regions(bad,:), neg_scores], 0.1)); % .3 is arbitrary
         all_inds = [all_inds; neg_inds(:)];
         cur_labels = [cur_labels; -ones(length(neg_inds), length(cls))];
      end
   end

   %cached{i} = [cached_scores{i}.part_scores(all_inds,:) all_inds(:)];
   if(isfield(cached_scores{i}, 'part_scores'))
      cached{i} = [cached_scores{i}.part_scores(all_inds,:) cached_scores{i}.region_score(all_inds,:) all_inds(:)];
   else
      cached{i} = [cached_scores{i}.region_score(all_inds,:) all_inds(:)];
   end

   labels{i} = cur_labels;%cached_scores{i}.labels(all_inds, :);
   scores{i} = cached_scores{i}.scores(all_inds);
   regions{i} = cached_scores{i}.regions(all_inds,:);
   imind{i} = repmat(i, length(all_inds), 1);
end

labels = cat(1, labels{:});
labels(labels>0) = 1;
labels(labels<0) = -1;
cached = cat(1, cached{:});
scores = cat(1, scores{:});
imind = cat(1, imind{:});

function best_ind = get_best_hyp(true_label, scores)

   [ind dk un_label] =  unique(true_label);

   for i = 1:length(ind)
      this_ind = find(un_label==i);
      [best_score best_ind_t] = max(scores(this_ind));
      best_ind(i) = this_ind(best_ind_t);
   end


