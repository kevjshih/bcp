function [Dsub cached_sub pos_inds neg_inds] = collect_pos_subset(cls, D, cached_scores, im_list, obj_box, best_part_boxes)

copy_neg = zeros(1, length(D));

[Dpos pos_inds0] = LMquery(D, 'object.name', cls, 'exact');
cached_pos = cached_scores(pos_inds0);

neg_inds = 1:length(D);
neg_inds(pos_inds0) = [];

pos_names = getfield2(Dpos, [], 'annotation', 'filename');
[good_names dk good_inds] = unique(im_list);


for i = 1:length(good_names)
   [dk goodname goodext] = fileparts(good_names{i});
   Dind = find(ismember(pos_names, [goodname goodext]));
   pos_inds(i) = Dind;
 
   good_ind = find(good_inds==i);
   good_boxes = cat(1, obj_box{good_ind});

   % Figure out which objects are ok
   boxes = LMobjectboundingbox(Dpos(Dind).annotation, cls);
   [overlaps best_ind] = max(bbox_overlap_mex(boxes, good_boxes), [], 2);
   good_obj = best_ind(overlaps>=0.5);

         
   % Delete all of the bad regions
   ok_regions = ismember(cached_pos{Dind}.labels, [-1; 0; good_obj]);
   cached_pos{Dind} = prune_cached_scores(cached_pos{Dind}, ok_regions);
end

Dsub = [D(pos_inds0(pos_inds)) D(neg_inds)];
cached_sub = [cached_pos(pos_inds) cached_scores(neg_inds)];
