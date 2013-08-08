function [labels cached imind scores boxes regions] = prune_boost_data_overlap(D, cached_scores, cls, loc)

INCLUDE_NEG = 1;
INCLUDE_LOC = 1; % Don't use this

%if(exist('loc','var') && loc==1)
%   INCLUDE_LOC = 1; % Don't use this
%else
%  INCLUDE_LOC = 0; % Don't use this
%end

for i = 1:length(cached_scores)
   all_inds = [];
   if(isempty(cached_scores{i}.labels))
       continue;
   end
   annotation = D(i).annotation;

   big_enough = ones(size(cached_scores{i}.scores)); %mean(isinf(cached_scores{i}.part_scores),2)<=1;%0.85;

   ok_pos = find(cached_scores{i}.labels>0 & big_enough);
   if(any(cached_scores{i}.labels>0)) % positive image
      boxes = LMobjectboundingbox(annotation, cls);
      [overlaps best_ind] = max(bbox_overlap_mex(boxes, cached_scores{i}.regions(ok_pos, :)), [], 2);

      pos_inds = ok_pos(best_ind(overlaps>0.5));
      all_inds = pos_inds;
   end 

%   if(INCLUDE_NEG)
%      bad = find(cached_scores{i}.labels<=0);
%   else
%      bad = find(cached_scores{i}.labels<0);
%   end 

   if(INCLUDE_LOC)
      bad_loc = find(cached_scores{i}.labels==0 & big_enough);
      if(any(bad_loc))
         annotation = D(i).annotation; 
         boxes = LMobjectboundingbox(annotation, cls);
         [overlaps] = max(bbox_overlap_mex(boxes, cached_scores{i}.regions(bad_loc, :)), [], 1);
         %bad_loc = bad_loc(overlaps<=0.5);
         bad_loc = bad_loc(overlaps<=0.35);
      end
      bad = [find(cached_scores{i}.labels<0 & big_enough); bad_loc(:)];
   else
      bad_loc = [];
      bad = [find(cached_scores{i}.labels<0 & big_enough); bad_loc(:)];
   end

   if(any(bad) && (1 || ~any(cached_scores{i}.labels>0)))
   %if(any(bad) && (INCLUDE_NEG || ~any(cached_scores{i}.labels>0)))
%   if(any(bad) && any(cached_scores{i}.labels>0)))
      if(~isempty(bad))
         neg_scores = bad; % Break ties with rank

         ROUGH_NMS = 1;
         if(ROUGH_NMS)
           [dk un] = unique(neg_scores);
            bad = bad(un);
           neg_inds = bad(rand(size(bad))<(25000/2e6));
%           neg_inds = bad(rand(size(bad))<(100000/2e6));
         else
           neg_inds = bad(nms_v4([cached_scores{i}.regions(bad,:), neg_scores], 0.5));
           %neg_inds = bad(nms_iou([cached_scores{i}.regions(bad,:), neg_scores], 0.1)); % .5 is arbitrary
         end
         all_inds = [all_inds; neg_inds(:)];
      end
   end

   %cached{i} = [cached_scores{i}.part_scores(all_inds,:) all_inds(:)];
   if(isfield(cached_scores{i}, 'part_scores'))
      cached{i} = [cached_scores{i}.part_scores(all_inds,:) cached_scores{i}.region_score(all_inds,:) all_inds(:)];
   else
      cached{i} = [cached_scores{i}.region_score(all_inds,:) all_inds(:)];
   end

   labels{i} = cached_scores{i}.labels(all_inds);
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


