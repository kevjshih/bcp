function [Dpos_split Dpos_inds cached_split aspect_ratio] = cluster_pos(D, cls, cached_scores, N)

[Dpos Dpos_inds0] = LMquery(D, 'object.name', cls, 'exact');

for i = 1:length(Dpos)
   % Find regions that have best overlap
   boxes0 = LMobjectboundingbox(Dpos(i).annotation, cls);

   % Get best overlapping region for each box
   [overlaps best_ind] = max(bbox_overlap_mex(boxes0, cached_scores{Dpos_inds0(i)}.regions), [], 2);

   gt_inds{i} = find(overlaps>=0.5);
   reg_inds{i} = best_ind(gt_inds{i});
   reg_box{i} = cached_scores{Dpos_inds0(i)}.regions(reg_inds{i},:);
   D_inds{i} = repmat(i, numel(gt_inds{i}), 1);
end

reg_boxes = cat(1, reg_box{:});
D_inds_all = cat(1, D_inds{:});
gt_inds_all = cat(1, gt_inds{:});
reg_inds_all = cat(1, reg_inds{:});

% From FGMR
h = [reg_boxes(:,4)]' - [reg_boxes(:,2)]' + 1;
w = [reg_boxes(:,3)]' - [reg_boxes(:,1)]' + 1;
aspects = h ./ w;
aspects = sort(aspects);

for i=1:N+1
  j = ceil((i-1)*length(aspects)/N)+1;
  if j > length(aspects)
    b(i) = inf;
  else
    b(i) = aspects(j);
  end
end

% Prepare for indexing hell!!
aspects = h ./ w;
for i=1:N
   I = find((aspects >= b(i)) .* (aspects < b(i+1)));
   [cur_D_inds dk Dmap] = unique(D_inds_all(I));
   cur_gt_inds = gt_inds_all(I);
   cur_reg_inds = reg_inds_all(I);

   for j = 1:length(cur_D_inds)
      D_ind = cur_D_inds(j);
      D0_ind = Dpos_inds0(D_ind);
      todo = Dmap==j;
      gt_ind = cur_gt_inds(todo);
      reg_ind = cur_reg_inds(todo);

      Dpos_split{i}(j).annotation = Dpos(D_ind).annotation;
      Dpos_split{i}(j).annotation.object = Dpos(D_ind).annotation.object(gt_ind);

      cached_split{i}{j}.regions     = cached_scores{D0_ind}.regions(reg_ind,:);
      cached_split{i}{j}.labels      = cached_scores{D0_ind}.labels(reg_ind,:);     
      cached_split{i}{j}.scores      = cached_scores{D0_ind}.scores(reg_ind);     
      cached_split{i}{j}.part_scores = cached_scores{D0_ind}.part_scores(reg_ind,:); 
      cached_split{i}{j}.part_boxes  = cached_scores{D0_ind}.part_boxes(reg_ind,:);  
      cached_split{i}{j}.region_score= cached_scores{D0_ind}.region_score(reg_ind,:);
   end

   Dpos_inds{i} = Dpos_inds0(cur_D_inds);

   aspect_ratio(i) = median(aspects(I));
end
