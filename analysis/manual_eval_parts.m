set = 'train';
cls = 'aeroplane';
parts_inds = 1:5;
rounds = 1:4;

email = 'jiaa1@illinois.edu';
subject = ['Evaluating \"' set '\" set ' cls ' parts'];

%% Retrieve the parts
all_parts = get_manual_parts(VOCopts, set, cls);
parts = {};
for parts_ind = parts_inds
   parts{end+1} = all_parts{parts_ind};
end

%% Select the applicability of the parts to the specified set
for parts_i = 1:length(parts)
   part = parts{parts_i};
   manual_set_applicability(VOCopts, part, 'test');
end

%% Initialize the evaluation for each desired round
for parts_i = 1:length(parts)
   part = parts{parts_i};
   for round = rounds
      manual_eval_manual_refine_part_round(VOCopts, part, round, true);
   end
   manual_eval_auto_refine_part(VOCopts, part, true);
end

%% Evaluate the part
message = 'Ready for input';
system(['echo ' message ' | mail -s "' subject '" ' email]);
for parts_i = 1:length(parts)
   part = parts{parts_i};
   for round = rounds
      manual_eval_manual_refine_part_round(VOCopts, part, round);
   end
   manual_eval_auto_refine_part(VOCopts, part);
end

%% Create Plots
for parts_i = 1:length(parts)
   part = parts{parts_i};

   recalls = [];
   precisions = [];
   curve_names = {};

   refinement_rounds = manual_refine_part(VOCopts, part, true);
   for round = rounds
      correct = refinement_rounds(round).eval_correct;

      % Compute recall and precision
      recall = zeros(length(correct), 1);
      precision = zeros(length(correct), 1);
      total_applicable = length(find(correct ~= -2));
      curr_total_correct = 0;
      for i = 1:length(correct)
         if correct(i) == 1
            curr_total_correct = curr_total_correct + 1;
         end
         recall(i) = curr_total_correct / total_applicable;
         precision(i) = curr_total_correct / i;
      end

      recalls = [recalls recall];
      precisions = [precisions precision];

      % Set curve name
      if round == 1
         curve_names{end+1} = '1 round';
      else
         curve_names{end+1} = [num2str(round) ' rounds'];
      end
      num_examples = length(find(refinement_rounds(round).refine_correct == 1));
      curve_names{end} = [curve_names{end} ' (' num2str(num_examples) ' examples)'];
   end

   correct = manual_eval_auto_refine_part(VOCopts, part, true);

   % Compute recall and precision
   recall = zeros(length(correct), 1);
   precision = zeros(length(correct), 1);
   total_applicable = length(find(correct ~= -2));
   curr_total_correct = 0;
   for i = 1:length(correct)
      if correct(i) == 1
         curr_total_correct = curr_total_correct + 1;
      end
      recall(i) = curr_total_correct / total_applicable;
      precision(i) = curr_total_correct / i;
   end

   recalls = [recalls recall];
   precisions = [precisions precision];

   % Set curve name
   curve_names{end+1} = 'automatic';

   % Show the part on the left subplot
   figure;
   subplot(1, 2, 1);
   imshow(part.icon);
   plot_bbox(part.icon_bbox);
   
   % Show the curves on the right subplot
   subplot(1, 2, 2);
   plot(recalls, precisions);
   legend(curve_names);
   xlabel('current amount of correct/total applicable');
   ylabel('current amount of correct/current amount of detections');
   title([strrep(part.name, '_', '\_') ' refinement evaluation on test set (class ' part.cls ')']);
end
