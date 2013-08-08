% I refactored the code out into 'manual_refine_part' and
% 'manual_eval_refine_part' but I didn't add those into this script yet.

cls_s = {'cat', 'aeroplane'};
set = 'train';

verify_applicability = true;

%% Load or select parts.
if fileexists('parts.mat')
   load('parts.mat', 'parts');
end

if ~exist('parts')
   % Get the parts for each class from the user.
   parts = {};
   for cls_s_i = 1:length(cls_s)
      cls_parts = ui_pascal_choose_parts(VOCopts, set, cls_s{cls_s_i});
      for cls_parts_i = 1:length(cls_parts)
         parts{end+1} = cls_parts{cls_parts_i};
      end
   end
   save('parts.mat', 'parts');
end

%% Precompute all the initialized part models.
parfor parts_i = 1:length(parts)
   part = parts{parts_i};
   cls = part.cls;
   orig_train_exemplar(VOCopts, part.im, part.bbox, cls, set, true);
end

return;

% Evaluate cats -- this is old code that doesn't work anymore
[model.part.computed] = deal(0);
[dc cached_scores_test] = collect_boost_data(model, D, cached_scores_test);

[ex_im ex_bbox im_list im_names obj_box obj_box0 best_part_boxes best_scores] = get_exemplar_boxes(Dtest, cached_scores_test, cls, model, 1);
correct = ui_check_detections(ex_im, ex_bbox, im_list, ...
                              obj_box, best_part_boxes, ...
                              best_scores);


clf;
plot(cumsum(correct)./(1:length(correct))')
hold on;
plot(cumsum(correct==1)./(cumsum(correct~=0)),'r')

legend('Manual All Obj', 'Manual Relevant Obj');


% Compare to automatic baseline
[model_auto.part.computed] = deal(0);
[dc cached_scores_test_auto] = collect_boost_data(model_auto, D, cached_scores_test);


%% Evaluate auto
load_init_data;

%% Set up model flags
model.hard_local = 0;
model.score_feat = 0;
model.incremental_feat = 0;
model.do_transform = 1;
model.shift = [0];
model.rotation = [0]; %[-20 -10 0 10 20]; % No shift for now, but
                      %we want
model.do_boxes = 1;

part_model = orig_train_exemplar(VOCopts, part.im, part.bbox, cls, set, true);
part_model.model.name = part_model{cls_parts_i}.models_name;

model = add_model(model, part_model.model);

%% Refine completely automatically.
[model dc w_loo w_noloo all_models] = train_loo_cache(model, D, cached_scores, 10, 2, 1, 15.0);

%% Evaluate the automatically-learned part on the test set.
[model.part.computed] = deal(0);
[dc cached_scores_test] = collect_boost_data(model_auto, D, cached_scores_test);
[dc dc im_list_test im_names_test obj_box_test obj_box0_test best_part_boxes_test best_scores_test] = get_exemplar_boxes(Dtest, cached_scores_test, cls, model, 1);

% TODO: fix code below
         part_applicability_test = internal_applicability(applicability_lookup_test, im_names_test, obj_box_test);
         correct_test = ui_check_part(part, im_list_test, obj_box_test, part_applicability_test, best_part_boxes_test, best_scores_test);

         % Save the data for the automatically-learned part performance.
         refinement_rounds_auto{parts_i}(1).im_names_test = im_names_test;
         refinement_rounds_auto{parts_i}(1).obj_box_test = obj_box_test;
         refinement_rounds_auto{parts_i}(1).best_part_boxes_test = best_part_boxes_test;
         refinement_rounds_auto{parts_i}(1).best_scores_test = best_scores_test;
         refinement_rounds_auto{parts_i}(1).correct_test = correct_test;

         model.part(cls_parts_i).computed = 1;
