function [pos_prec chosen aps] = choose_candidate_amp(model, D, cached_scores, candidates)
 
cls = model.cls;

% Temporary fix, remove any elements of D that don't have regions
%remove = zeros(numel(D), 1);
%for i = 1:length(D)
%   if(isempty(cached_scores{i}.regions))
%      remove(i) = 1;
%   end
%end

% 2) Load score data
num_parts = numel(candidates);

[dk pos_inds] = LMquery(D, 'object.name', cls, 'exact');
neg_inds = 1:length(D);
neg_inds(pos_inds) = [];
neg_inds = neg_inds(1:min(end, max(200, length(pos_inds))));

D = D([pos_inds(:); neg_inds(:)]);
cached_scores = cached_scores([pos_inds(:); neg_inds(:)]);

cached_gt = get_gt_pos_reg(D, cached_scores, cls);


% Collect positive precisions
for i = 1:length(candidates)
   %fprintf('%d/%d\n', i, num_parts);
   [aps(i) pos_prec{i}] = test_part_detections_wrapper(D, cached_gt, cls, candidates(i));
   fprintf('%f\n', aps(i));
end

ann = [D.annotation];
Dnames = {ann.filename};

% Remove examples that were used for training when computing max ap
% (Otherwise we could do perfectly on N objects by selecting N parts, which wouldn't hold on the test set)
for i = 1:length(candidates)
   fn = regexp(candidates{i}.name, '\d{4}_\d{6}', 'match');
   to_rem = strcmp(Dnames, [fn{1} '.jpg']);
   pos_prec_test{i} = pos_prec{i};
   if(any(to_rem))
    pos_prec_test{i}{to_rem} = zeros(size(pos_prec{i}{to_rem}));
   end
end


% put them in matrix form
pos_prec_mat = [];
for i = 1:length(pos_prec)
   pos_prec_mat = [pos_prec_mat, cat(2, pos_prec_test{i}{:})'];
end


% Greedy forward selection of average max ap
current_max = zeros(size(pos_prec_mat,1), 1);
gain = zeros(numel(pos_prec), 1);

current_max = zeros(size(pos_prec_mat,1), 1);

for i = 1:length(pos_prec)
   % choose 
   to_gain = mean(max(0, bsxfun(@minus, pos_prec_mat, current_max)));

   [gain(i) chosen(i)] = max(to_gain);

   current_max = max(current_max, pos_prec_mat(:,chosen(i)));
   if(gain(i)==0)
      fprintf('no more parts help, terminating!\n');
      break;
   end
end


return;

if(0)
% Do many trials
Nto_use = 50;
gain = zeros(1000, nto_use);
for trial = 1:100
   trial
   current_max = zeros(size(pos_prec_mat,1), 1);
   r = randperm(length(pos_prec));
   touse = r(1:nto_use);
   sub_mat = pos_prec_mat(:, touse);

   for i = 1:nto_use%length(pos_prec)
      % choose 
      to_gain = mean(max(0, bsxfun(@minus, sub_mat, current_max)));

      [gain(trial, i) chosen(trial, i)] = max(to_gain);

      current_max = max(current_max, sub_mat(:,chosen(trial, i)));
%      plot(sort(current_max));
      if(gain(trial,i)==0)
         fprintf('no more parts help, terminating!\n');
         break;
      end
   end
   errorbar(1:nto_use, mean(cumsum(gain(1:trial,:),2),1), std(cumsum(gain(1:trial,:),2),0,1))
   drawnow
%   pause
end
end

function [aps pos_prec] = test_part_detections_wrapper(D, cached_scores, cls, candidate);
   results = test_candidate_detections(D, cached_scores, cls, candidate);

   if(~isempty(results{1}))
      full_results = results{1};
      nneg = max(200, length(full_results)/2);
      npos = length(full_results) - nneg;

      [recall prec aps pos_prec] = test_part_detections_D(cls, D, cached_scores, full_results, (5000-npos)/nneg);
   else
      aps = 0;
      pos_prec = [];
   end
