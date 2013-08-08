function show_top_parts(D, cached_scores, cls, model)
% models = load_candidate_models(cls);
% :w:
im_dir = [];
BGglobals;

[D inds] = LMquery(D, 'object.name', cls, 'exact');
cached_scores = cached_scores(inds);

part_scores = {};
part_boxes = {};

parfor i = 1:length(D)
   fprintf('%d/%d\n', i, length(D));
   im = imread(fullfile(im_dir, D(i).annotation.filename));
   [part_scores{i} part_boxes{i}] = part_inference(im, model, cached_scores{i}.regions);
end

for i = 1:length(cached_scores)
   cached_scores{i}.part_scores = part_scores{i}{1};
   cached_scores{i}.part_boxes = part_boxes{i}{1};
   cached_scores{i}.part_scores(cached_scores{i}.labels<=0) = -inf;
end

show_top_parts(D, cached_scores, cls, 1);
