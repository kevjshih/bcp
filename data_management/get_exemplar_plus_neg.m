function [Dsub cached_sub] = get_exemplar_plus_neg(D, cached_scores, part_model, cls)

[exim exbox] = extract_exemplar_params(part_model);

fns = getfield2(D, [], 'annotation', 'filename');
ok_pos = find(ismember(fns, exim));

Dpos = D(ok_pos);
cached_pos = cached_scores{ok_pos};

boxes = LMobjectboundingbox(Dpos.annotation, cls);
[dk ex_ind] =  max(bbox_overlap(boxes, exbox),[],2);

ok = cached_pos.labels==ex_ind;
cached_pos.regions = cached_pos.regions(ok, :);
cached_pos.labels = cached_pos.labels(ok);
cached_pos.scores = cached_pos.scores(ok);
cached_pos.part_scores = cached_pos.part_scores(ok, :);
cached_pos.part_boxes = cached_pos.part_boxes(ok, :);
cached_pos.region_score = cached_pos.region_score(ok, :);

[dk inds] = LMquery(D, 'object.name', cls, 'exact');
Dsub = D;
Dsub(inds) = [];
Dsub = [Dpos Dsub];

cached_sub = cached_scores;
cached_sub(inds) = [];
cached_sub = [{cached_pos} cached_sub];

