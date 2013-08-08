function characterize_overlaps(D, cached_scores, model, part_todo)

cls = model.cls;

[model.part.computed] = deal(1);

model.part(part_todo).spat_const = [0 1 0.8 1 0 1];
model.part(part_todo).reference_box = [];
model.part(part_todo).computed = 0;

[Dpos pos_inds] = LMquery(D, 'object.name', cls, 'exact');
cached_pos = cached_scores(pos_inds);

cached_gt = get_gt_pos_reg(Dpos, cached_pos, cls);

[dk pos_hyp] = collect_training_ex(model, Dpos, cached_gt, 1);


gt_box = {};
part_box = {};
part_scores = {};
region_boxes = {};

for i = 1:length(pos_hyp)
   hyps = pos_hyp{i};
   pos_reg = find(cached_gt{i}.labels>0);
   for h = 1:length(hyps)
      gt_box{end+1} = cached_gt{i}.regions(pos_reg(hyps(h).region), :);
      part_box{end+1} = hyps(h).bbox(part_todo, :);
      part_scores{end+1} = hyps(h).score(part_todo);
      ok_regions = bbox_overlap_mex(cached_pos{i}.regions, gt_box{end})>=0.5;
      region_boxes{end+1} = cached_pos{i}.regions(ok_regions, :);
   end
end

IoverR = zeros(length(part_box), 2);
IoverP = zeros(length(part_box), 2);
IoverU = zeros(length(part_box), 2);

for i = 1:length(part_box)
   if(~isempty(region_boxes{i}));
      IoR = bbox_contained(region_boxes{i}, part_box{i});
      %IoR = bbox_contained(gt_box{i}, part_box{i}, 0);
      IoverR(i,:) = [min(IoR) max(IoR)];

      IoP = bbox_contained(part_box{i}, region_boxes{i});
      %IoP = bbox_contained(part_box{i}, gt_box{i}, 0);
      IoverP(i, :) = [min(IoP), max(IoP)];

      IoU = bbox_overlap_mex(part_box{i}, region_boxes{i});
      %IoU = bbox_overlap_mex(part_box{i}, gt_box{i});
      IoverU(i, :) = [min(IoU), max(IoU)];
   end
end

scores = cat(1, part_scores{:});

[scores_sort sort_ind] = sort(scores, 'ascend');

