function [model w_loo w_orig all_models neg_feats] = train_boosted(model, D, cached_scores, num_iter, Nouter, rate, C, neg_feats)

if(~isfield(model, 'do_loo'))
   do_loo = 1;
else
   do_loo = model.do_loo;
end

if(nargout==1)
   do_loo = 0;
end

if(~exist('num_iter', 'var'))
    num_iter = Inf;
end
    

if(~exist('Nouter', 'var'))
   Nouter = 10;
end

if(~exist('rate', 'var'))
   rate = 1;
end

blocksize = 200;
cls = model.cls;
% Setup data
[dk inds] = LMquery(D, 'object.name', cls, 'exact');
Dpos = D(inds);
cached_pos = cached_scores(inds);

% This makes sure we only worry about positive regions
for i = 1:length(cached_pos)
   cached_pos{i} = prune_cached_scores(cached_pos{i}, cached_pos{i}.labels>0);
end

Dneg = D;
cached_neg = cached_scores;

if(~isfield(model, 'hard_local') || model.hard_local==0)
   Dneg(inds) = [];
   cached_neg(inds) = [];
end

N = length(Dneg);

blocksize = min(blocksize, N);
%Nouter = 10;

if(~exist('neg_feats', 'var'))
   neg_feats = cell(N,1);
elseif(numel(neg_feats)~=N)
   error('Previous negative feature set doesn''t match current negative training set!\n');
end

all_models = {};


if(isfield(model, 'score_feat') && model.score_feat==0)
   NO_SCORE_FEAT = 1;
else
   NO_SCORE_FEAT = 0;
end

if(NO_SCORE_FEAT)
   model.cached_weight(:) = 0;
else
   %model.cached_weight = 10; % Start by emphasizing previous results
   model.cached_weight(:) = 0;
end

if(~exist('C','var'))
   C = 5;
end

w = get_model_weights(model);

bookkeeping = cell(1, numel(Dneg));

subset_split_size = model.subset_split;
model.subset_split = 0; % Start with no split to get initialization

for outer_iter = 1:Nouter
   if(outer_iter==3) % Time to use every example again!
      cached_pos = clear_split_labels(cached_pos);
      keyboard
   end
 
   [pos_feats0 pos_hyp] = collect_positives(model, Dpos, cached_pos);
   pos_feats = cat(2, pos_feats0{:});
   Npos = size(pos_feats,2);

   if(outer_iter==1) 
      % Begin by selecting the best and worst subsets
      w0 = get_model_weights(model);
      pred = w0(1:end-1)'*pos_feats + w0(end);
      [a b] = sort(pred, 'descend');

      Kn = ceil(numel(pred)*rate);
      ub = a(Kn); % anything greater than ub is in positive set
      lb = a(end-Kn); % anything less than lb is in contrast set
      
      cached_pos = update_split_labels(cls, Dpos, cached_pos, pos_hyp, ub, lb);
     % pos_feats0 = update_split_features(pos_feats0, pos_hyp, ub, lb);
      % Rerun inference
      model.subset_split = subset_split_size;
      model = add_subset_model(model);
      w = get_model_weights(model); % Get new weights!
      fprintf('Rerunning positive inference with split subset model!\n');
      [pos_feats0 pos_hyp] = collect_positives(model, Dpos, cached_pos);
      pos_feats = cat(2, pos_feats0{:});
      Npos = size(pos_feats,2);
   end
   %Cf = C/Npos;

   r = randperm(N);

   inner_iter = 0;
   obj_prev = -inf;
   obj = 0;

   %while(inner_iter<num_iter)% && abs(obj-obj_prev)/obj>0.005) <-- This only works when you loop over all images 
   for inner_iter = 1:num_iter%min(num_iter, ceil(N/blocksize))
      obj_prev = obj;

      %first = (inner_iter-1)*blocksize+1;
      %last = min(inner_iter*blocksize, N);
      block = unique(sort(ceil(rand(blocksize,1)*N)));%sort(r(first:last));

      %neg_feats(block) = collect_negatives(model, Dneg(block), cached_neg(block));
      new_neg = collect_negatives(model, Dneg(block), cached_neg(block));
      [neg_feats bookkeeping] = update_neg_cache(neg_feats, bookkeeping, new_neg, block, w);

      % Update model
      feats = [pos_feats, cat(2, neg_feats{:})];
      labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];

      if(NO_SCORE_FEAT)
         feats(end,:) = 0; % Don't use the score of previous iterations
      end
      
      fprintf('Iter: %d/%d  - %d/%d\n', outer_iter, Nouter, inner_iter, ceil(N/blocksize));
      w0 = get_model_weights(model);

      if(1 || inner_iter==1)
         Kn = ceil(sum(labels==1)*1);% rate); % No more subset here!!
         Cf = C/Kn;
         pred = w0(1:end-1)'*pos_feats;

         [a b] = sort(pred, 'descend');
         delta0 = ones(length(labels), 1);
         delta0(1:size(pos_feats,2)) = 0;
         delta0(b(1:min(end,Kn))) = 1; % Choose 


         [w deltas alphas obj] = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*1), delta0);
         delta_pos = deltas(1:size(pos_feats,2));
      else % Use previous deltas to speed things up
         new_deltas = ones(numel(labels), 1);
         new_deltas(labels==1) = delta_pos;
         [w deltas alphas] = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*1), new_deltas);
         delta_pos = deltas(1:size(pos_feats,2));
      end
      clear feats; 
      all_models{end+1} = model;
      model = update_model_weights(model, w);
   end
end

% Compute which image each positive feature came from
for i = 1:length(pos_hyp)
   pos_im_ind{i} = repmat(i, length(pos_hyp{i}),1);
end

pos_im_inds = cat(1, pos_im_ind{:});

% Given final deltas, recompute w for examples that were included
pos_to_do = find(deltas(1:length(pos_im_inds)));

pos_im_todo = unique(pos_im_inds(pos_to_do));

% Given final deltas, recompute w for examples that were included
delta_pos = deltas(1:length(pos_im_inds));
delta_neg = deltas(length(pos_im_inds)+1:end);

% 
all_alphas = zeros(length(deltas), 1);
all_alphas(deltas==1) = alphas;

alpha_pos = all_alphas(1:length(pos_im_inds));
alpha_neg = all_alphas(length(pos_im_inds)+1:end);

w_loo0 = cell(1,length(pos_im_todo));
fast_subset = 1;

w_orig = w;

if(do_loo)
   % TODO: Put a parfor here!
   for i = 1:length(pos_im_todo)
      fprintf('Computing leave one out estimate: %d/%d\n', i, length(pos_im_todo));
      curim = pos_im_todo(i);
   
      delta1 = [delta_pos(pos_im_inds~=curim); delta_neg];
      alpha0 = [alpha_pos(pos_im_inds~=curim); alpha_neg];
   
      pos_feats = cat(2, pos_feats0{[1:curim-1, curim+1:end]});
      feats = [pos_feats, cat(2, neg_feats{:})];
      if(NO_SCORE_FEAT)
         feats(end,:) = 0; % Don't use the score of previous iterations
      end
      
      labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];
   
      %w_loo0{i} = svm_dual_mex(labels(delta1==1), feats(:, delta1==1), C)';
      if(numel(Cf)==1)
         w_loo0{i} = svm_dual_mex(labels(delta1==1), feats(:, delta1==1), Cf, [], alpha0)'; % Initialize using previous alpha
      else
         [wt b] = fast_svm(labels(delta1==1), feats(:, delta1==1), Cf(1), Cf(2)/Cf(1));
         w_loo0{i} = [wt(:); b];
      end
   end

   w_loo = cell(1, length(pos_feats0));
   w_loo(pos_im_todo) = w_loo0;
   for i = 1:length(pos_feats0)
      if(~ismember(i, pos_im_todo))
         w_loo{i} = w;
      end
   end
else % don't do loo, just copy over model
   w_loo = cell(1, length(pos_feats0));
   for i = 1:length(pos_feats0)
      w_loo{i} = w;
   end
end



function [neg_feats neg_bookkeep] = update_neg_cache(neg_feats, neg_bookkeep, new_neg, block, w)

% If example hasn't been inside margin for 3 iterations

% First, add new negatives to existing set, and compute scores
num_dup = 0;
num_dup1 = 0;
for i = 1:length(block)
   bl = block(i);
   % neg_to_keep = neg_feats{bl}(:, ~find_duplicates1(neg_feats{bl}));
   neg_to_add = new_neg{i}(:, ~find_duplicates1(new_neg{i}));
   neg_to_add = neg_to_add(:, ~find_duplicates(neg_feats{bl}, neg_to_add));

   num_dup = num_dup + size(new_neg{i},2) - size(neg_to_add,2);
%   num_dup1 = num_dup1 + size(neg_feats{bl},2) - size(neg_to_keep,2);

   if(any(num_dup1))
      keyboard;
   end
   neg_feats{bl} = [neg_feats{bl} neg_to_add];
   % Count the number of times outside the margin
   neg_bookkeep{bl} = [neg_bookkeep{bl}, zeros(1,size(neg_to_add,2))];
end

scores = cell(numel(neg_feats),1);

% Next, Compute scores
for i = 1:numel(neg_feats)
   if(~isempty(neg_feats{i}))
    scores{i} = w(1:end-1)'*neg_feats{i} + w(end);
   end
end

% Find thresh
sorted_scores = sort(cat(2,scores{:}), 'descend');
sorted_scores(isinf(sorted_scores)) = [];

if(isempty(sorted_scores))
   thresh = -inf;
else
   thresh = sorted_scores(min(end,10000));
end

fprintf('Number of duplicates: %d (%f)\n', num_dup, num_dup/(numel(sorted_scores)-num_dup));
%fprintf('Number of concurrent duplicates: %d\n', num_dup1);

if(thresh>-1)
   fprintf('Warning: Not all support vectors are being kept!\n');
end

for i = 1:length(neg_feats)
   % Increment everything
   neg_bookkeep{i} = neg_bookkeep{i}+1;
   
   % Reset any that fall in (or very close to) margin
   neg_bookkeep{i}(scores{i}>=-1.01) = 0;
   to_remove = neg_bookkeep{i}(:)>=2 & scores{i}(:)<thresh;
   neg_feats{i}(:, to_remove) = [];
   neg_bookkeep{i}(to_remove) = [];
end


function new_dup = find_duplicates1(existing)

% Compute 
vals0 = sum(existing.^2, 1);
new_dup = zeros(size(existing,2),1);

for i = 1:length(vals0)
   tocheck = abs(vals0(i)-vals0)<1e-9;
   tocheck(1:i) = 0; % Don't want to remove first occurrence

   if(any(tocheck))
      same = abs(vals0(i) - existing(:,i)'*existing(:, tocheck))<1e-9;

      new_dup(tocheck) = reshape(new_dup(tocheck), [], 1) | same(:);
   end
end


function new_dup = find_duplicates(existing, new_ex)

% Compute 
vals0 = sum(existing.^2, 1);
valsN = sum(new_ex.^2, 1);

new_dup = zeros(size(new_ex,2),1);

for i = 1:length(vals0)
   tocheck = abs(vals0(i)-valsN)<1e-9;
   
   if(any(tocheck))
      same = abs(vals0(i) - existing(:,i)'*new_ex(:, tocheck))<1e-9;

      %new_dup(tocheck) = new_dup(tocheck) | same;
      new_dup(tocheck) = reshape(new_dup(tocheck), [], 1) | same(:);
   end
end

function cached_pos = clear_split_labels(cached_pos)

   for i = 1:length(cached_pos)
      cached_pos{i}.regions(:, 5) = zeros(size(cached_pos{i}.regions,1), 1);
   end

function cached_pos = update_split_labels(cls, Dpos, cached_pos, pos_hyp, ub, lb)

for i = 1:length(Dpos)
   h = pos_hyp{i};
   if(isempty(h))
      continue;
   end
   gt_boxes = LMobjectboundingbox(Dpos(i).annotation, cls);
   gt_labels = -ones(size(gt_boxes,1), 1);
   for j = 1:length(h)
      % Figure out which gt region this hyp came from
      [a b] = max(bbox_overlap_mex(cached_pos{i}.regions(h(j).region,:), gt_boxes));

      if(h(j).final_score>ub)
         gt_labels(b) = 1; % Use subset model
      elseif(h(j).score<lb)
         gt_labels(b) = 2; % Use alternate model
      end
   end

   cached_pos{i}.regions(:,5) = gt_labels(cached_pos{i}.labels);   
end
