function combine_part_models_test()
% Tests if combine_part_models is correct.

part_models = {};
model.num_parts = 2;
model.part(1).name = '1';
model.part(2).name = '2';
part_models{end+1} = model;

model.num_parts = 1;
model.part(1).name = '3';
part_models{end+1} = model;

model.num_parts = 1;
model.part(1).name = '4';
part_models{end+1} = model;

part_cached_scores = {};
cached_scores{1}.part_scores = ones(10, 1);
cached_scores{1}.part_boxes = ones(10, 4);
cached_scores{2}.part_scores = zeros(10, 1);
cached_scores{2}.part_boxes = zeros(10, 4);
part_cached_scores{end+1} = cached_scores;

cached_scores{1}.part_scores = zeros(10, 1);
cached_scores{1}.part_boxes = zeros(10, 4);
cached_scores{2}.part_scores = ones(10, 1);
cached_scores{2}.part_boxes = ones(10, 4);
part_cached_scores{end+1} = cached_scores;

expected_cached_scores = {};
expected_cached_scores{1}.part_scores = [ones(10, 1) zeros(10, 1)];
expected_cached_scores{1}.part_boxes = [ones(10, 4) zeros(10, 4)];
expected_cached_scores{2}.part_scores = [zeros(10, 1) ones(10, 1)];
expected_cached_scores{2}.part_boxes = [zeros(10, 4) ones(10, 4)];

[actual_model, actual_cached_scores] = combine_part_models(part_models, part_cached_scores);

assert(length(expected_cached_scores) == length(actual_cached_scores));
assert(all(all(expected_cached_scores{1}.part_scores == actual_cached_scores{1}.part_scores)));
assert(all(all(expected_cached_scores{1}.part_boxes == actual_cached_scores{1}.part_boxes)));
assert(all(all(expected_cached_scores{2}.part_scores == actual_cached_scores{2}.part_scores)));
assert(all(all(expected_cached_scores{2}.part_boxes == actual_cached_scores{2}.part_boxes)));

assert(4 == actual_model.num_parts);
assert('1' == actual_model.part(1).name);
assert('2' == actual_model.part(2).name);
assert('3' == actual_model.part(3).name);
assert('4' == actual_model.part(4).name);
disp('Success');
end