function [Dsub cached_sub pos_ind neg_ind] = collect_pos_subset(cls, ...
                                                  D, cached_scores, pos_prec, thresh)
% creates a temporary cached_scores structure that removes any
% positive example that has lower than 'thresh' precision for the
% current part

copy_neg = zeros(1, length(D));

Dpos = [];
cached_pos = {};

neg_ind = [];
pos_ind = [];

for i = 1:length(D)
   if(~isempty(cached_scores{i}) && ~any(cached_scores{i}.labels>=0)) % Negative example, we'll just copy it over later
      if(any(strcmp({D(i).annotation.object.name}, cls)))
         fprintf('Skipping a missed example\n');
         continue;
      end
      copy_neg(i) = 1;
      neg_ind(i) = i;
   elseif(i<=length(pos_prec)) % Positive example, include regions
      good_obj = find(pos_prec{i}>thresh);
       
      if(numel(good_obj)>0)
%         objs = D(i).annotation.object;
%         objs = objs(strcmp({objs.name}, cls));

         pos_ind(end+1) = i;
         Dpos(end+1).annotation = D(i).annotation;
%         Dpos(end).annotation.object = objs(good_obj);

         cached_pos{end+1} = cached_scores{i};
         
         % Delete all of the bad regions
         ok_regions = ismember(cached_pos{end}.labels, [-1 0 good_obj]);
         cached_pos{end} = prune_cached_scores(cached_pos{end}, ok_regions);
         
         %cached_pos{end}.regions(~ok_regions,:) = [];
         %cached_pos{end}.labels(~ok_regions) = [];
         %cached_pos{end}.scores(~ok_regions) = [];
         %cached_pos{end}.part_boxes(~ok_regions,:) = [];
         %cached_pos{end}.region_score(~ok_regions,:) = [];
      end
   end
end

Dsub = [Dpos D(copy_neg==1)];
cached_sub = [cached_pos cached_scores(copy_neg==1)];
