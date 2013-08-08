function [next_pos_prec next_chosen next_ap] = choose_candidate_amp(model, D, cached_scores, candidates, pos_prec)

cls = model.cls;

[cur_ap cur_pos_prec missed] = test_detections_wrapper(D, cached_scores, cls);

ann = [D.annotation];
Dnames = {ann.filename};

% Remove examples that were used for training when computing max ap
% (Otherwise we could do perfectly on N objects by selecting N parts, which wouldn't hold on the test set)
for i = 1:length(candidates)
   fn = regexp(candidates{i}.name, '\d{4}_\d{6}', 'match');
   to_rem = strcmp(Dnames, [fn{1} '.jpg']);
   
   if(~isempty(to_rem))
      pos_prec_test{i} = pos_prec{i};
      pos_prec_test{i}{to_rem} = zeros(size(pos_prec{i}{to_rem}));
   end
end

all_missed = cat(2, missed{:});

% put them in matrix form
pos_prec_mat = [];
for i = 1:length(pos_prec)
   pos_prec_mat = [pos_prec_mat, cat(2, pos_prec_test{i}{:})'];
end

pos_prec_mat = pos_prec_mat(~all_missed, :);

% Greedy forward selection of average max ap
%gain = zeros(numel(pos_prec), 1);
current_max = cat(2, cur_pos_prec{:})';
current_max = current_max(~all_missed, :);

%for i = 1:length(pos_prec)
i = 1;
to_gain = mean(max(0, bsxfun(@minus, pos_prec_mat, current_max)));

[gain(i) chosen(i)] = max(to_gain);

if(gain==0)
   fprintf('Warning!! No more gain to be had!\n');
else
   fprintf('Expected minimum gain: %f\n', gain);
end

current_max = max(current_max, pos_prec_mat(:,chosen(i)));

next_chosen = chosen;
next_pos_prec = pos_prec{chosen};

return;

function [aps pos_prec missed] = test_detections_wrapper(D, cached_scores, cls, candidate);
%   results = test_candidate_detections(D, cached_scores, cls, candidate);
   for i = 1:length(cached_scores)
      full_results{i} = cached_scores{i}.scores;
   end

   [recall prec aps pos_prec] = test_part_detections_D(cls, D, cached_scores, full_results);
   
   for i = 1:length(pos_prec)
      missed{i} = pos_prec{i}==0;
   end

