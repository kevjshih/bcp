function [model w_loo w_orig all_models neg_feats] = train_boosted(model, D, cached_scores, num_iter, Nouter, rate, C, neg_feats)
% This also does loo estimates on negatives

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

pos_inds = inds;
neg_inds = 1:length(D);

Dneg = D;
cached_neg = cached_scores;

if(~isfield(model, 'hard_local') || model.hard_local==0)
   Dneg(inds) = [];
   cached_neg(inds) = [];
   neg_inds(inds) = [];
else
   error('Not currently designed to handle hard localization errors (plus it just makes things worse)!\n');
end

N = length(Dneg);
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
   model.cached_weight = 0;
else
   model.cached_weight = 10; % Start by emphasizing previous results
end

if(~exist('C','var'))
   C = 1e-2;
end

w = get_model_weights(model);

for outer_iter = 1:Nouter
   [pos_feats0 pos_hyp] = collect_positives(model, Dpos, cached_pos);
   pos_feats = cat(2, pos_feats0{:});
   Npos = size(pos_feats,2);
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
      [neg_feats] = update_neg_cache(neg_feats, [], new_neg, block, w);

      % Update model
      feats = [pos_feats, cat(2, neg_feats{:})];
      labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];

      if(NO_SCORE_FEAT)
         feats(end,:) = 0; % Don't use the score of previous iterations
      end
      
      fprintf('Iter: %d/%d  - %d/%d\n', outer_iter, Nouter, inner_iter, ceil(N/blocksize));
      %[w deltas alphas] = subset_svm(feats, labels, 1e-2, 288*rate, reg);
      w0 = get_model_weights(model);

      if(1 || inner_iter==1)
         [w deltas alphas obj] = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*rate));
         delta_pos = deltas(1:size(pos_feats,2));
      else % Use previous deltas to speed things up
         new_deltas = ones(numel(labels), 1);
         new_deltas(labels==1) = delta_pos;
         [w deltas alphas] = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*rate), new_deltas);
         delta_pos = deltas(1:size(pos_feats,2));
      end
      clear feats; 
      all_models{end+1} = model;
      model = update_model_weights(model, w);
   end
end

all_feats = [pos_feats0'; neg_feats];

% Compute which image each feature came from
for i = 1:length(all_feats)
   im_ind{i} = repmat(i, size(all_feats{i},2), 1);
end

im_inds = cat(1, im_ind{:});
im_lab = [ones(length(pos_feats0),1); -ones(length(neg_feats),1)];

% Given final deltas, recompute w for examples that were included
ex_to_do = find(deltas==1 & alphas>1e-7);
im_todo = unique(im_inds(ex_to_do));

fast_subset = 1;

w_orig = w;

if(do_loo)
   w_loo = repmat({w_orig}, length(D), 1);
   
   w_loo0 = cell(1,length(im_todo));

   for i = 1:length(im_todo)
      curim = im_todo(i);
   
      fprintf('Computing leave one out estimate: %d/%d\n', i, length(im_todo));
      delta1 = deltas(im_inds~=curim);
      alpha0 = alphas(im_inds~=curim);
      feats = cat(2, all_feats{[1:curim-1, curim+1:end]});

      if(NO_SCORE_FEAT)
         feats(end,:) = 0; % Don't use the score of previous iterations
      end

      labels = im_lab(im_inds(im_inds~=curim));
   
      if(fast_subset) % Just train on precomputed subset
         w_loo0{i} = svm_dual_mex(labels(delta1==1), feats(:, delta1==1), Cf, [], alpha0)'; 
      else
         w_loo0{i} = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*rate));
      end
   
   end % loop over imset
   
   % Match loo model for each example, ordered as [pos ... neg]
   w_loo(im_todo) = w_loo0;


   % Now reorder to match the ordering of the input D
   w_loo([pos_inds(:); neg_inds(:)]) = w_loo;
end


function [neg_feats neg_bookkeep] = update_neg_cache(neg_feats, neg_bookkeep, new_neg, block, w)

% If example hasn't been inside margin for 3 iterations

% First, add new negatives to existing set, and compute scores
num_dup = 0;
num_dup1 = 0;
for i = 1:length(block)
   bl = block(i);
   neg_to_keep = neg_feats{bl}(:, ~find_duplicates1(neg_feats{bl}));
   neg_to_add = new_neg{i}(:, ~find_duplicates(neg_to_keep, new_neg{i}));

   num_dup = num_dup + size(new_neg{i},2) - size(neg_to_add,2);
   num_dup1 = num_dup1 + size(neg_feats{bl},2) - size(neg_to_keep,2);
   neg_feats{bl} = [neg_to_keep neg_to_add];
%   neg_bookkeep{bl} = [neg_bookkeep{bl}, zeros(1,size(neg_to_add,2))];
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
fprintf('Number of concurrent duplicates: %d\n', num_dup1);

if(thresh>-1)
   fprintf('Warning: Not all support vectors are being kept!\n');
end

for i = 1:length(neg_feats)
   neg_feats{i}(:, scores{i}<thresh) = [];
%   neg_bookkeep{bl} = [neg_bookkeep{bl}, zeros(1,size(neg_to_add,2))];
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

