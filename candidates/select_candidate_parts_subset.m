function [best_model best_ind ap] = select_candidate_parts(model, D, cached_scores, candidates, subset, weighted)

if(~exist('weighted','var'))
   weighted = 1;
end

cls = model.cls;
%%% Outline %%%%
% 1) select regions using previous scores %%
%    e.g. using nms/prune regions %
% 2) compute weighted AP for each part
% 3) choose best AP

% Temporary fix, remove any elements of D that don't have regions
%remove = zeros(numel(D), 1);
%for i = 1:length(D)
%   if(isempty(cached_scores{i}.regions))
%      remove(i) = 1;
%   end
%end

%D = D(~remove);
%cached_scores = cached_scores(~remove);

% 1) Do NMS
fprintf('Pruning regions\n');
[Dind regind labels weights] = prune_regions(D, cached_scores);
all_labels = cat(1, labels{:});
all_bin_labels = 2*double(all_labels>0)-1;
all_weights = cat(1, weights{:});

if(~weighted)
   all_weights = ones(size(all_weights));
end

weight_Z = 1/sum(all_weights);

% 2) Load score data
num_parts = numel(candidates);

num_reg = 0;
for i = 1:length(regind)
   all_regind{i} = num_reg + regind{i};
   num_reg = num_reg + numel(cached_scores{i}.labels);
end
all_regind = cat(1, all_regind{:});

start = tic;
for i = 1:num_parts
   if(toc(start)>5)
      fprintf('%d/%d\n', i, num_parts);
      start = tic;
   end
   results = test_candidate_detections(D, cached_scores, cls, candidates(i));
   if(~isempty(results{1}))
      all_res = cat(1, results{1}{:});
      all_scores = all_res(all_regind);

%      tic;
%      part_scores = apply_pruning(regind, results{1});
%      all_scores = cat(1, part_scores{:});
%      toc;
      [dk ordering] = sort(all_scores, 'descend');
      [un_label pos_ranks] = unique(all_labels(ordering), 'first');
      pos_labels = un_label(un_label>0);
      pos_ranks = pos_ranks(un_label>0);
      [pos_ordering] = sort(pos_ranks, 'ascend');

      top_pos = pos_labels(pos_ranks<=pos_ordering(ceil(end*subset)));
      touse = ismember(all_labels, [-1; 0; top_pos]);

      roc(i) = computeROCw_dup(all_scores(touse), all_labels(touse), all_weights(touse));
      ap(i) = VOCap(roc(i).r, roc(i).p);
   else
      ap(i) = 0;
   end
end

[dk best_ind] = max(ap);
best_model = candidates{best_ind};


function [scores] = apply_pruning(regind, results)

for i = 1:length(regind)
   scores{i} = results{i}(regind{i});
end

function [Dind regind label weights] = prune_regions(D, cached_scores)

for i = 1:length(D)
   if(isempty(cached_scores{i}.regions))
       continue;
   end
  
   inds = nms_v4([cached_scores{i}.regions, cached_scores{i}.scores(:)], 0.5);

   Dind(i) = i;%repmat(i, numel(inds), 1);
   regind{i} = inds(:);
   label{i} = cached_scores{i}.labels(inds);
   lab_tmp = 2*double(cached_scores{i}.labels(inds)>0)-1;
   weights{i} = 1./(1+exp(lab_tmp.*cached_scores{i}.scores(inds)));
end

num_pos = 0;
% Assign unique labels
for i = 1:length(D)
   pos_ind = label{i}>0;

   if(any(pos_ind))
      pos_label = label{i}(pos_ind);
   
      [un_pos dk un_inds] = unique(pos_label);
      pos_map = un_inds + num_pos;
   
      label{i}(pos_ind) = un_inds + num_pos;
      num_pos = num_pos + length(un_pos);
   end
end
