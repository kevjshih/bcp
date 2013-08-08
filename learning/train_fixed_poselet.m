function [model neg_feats w_loo w_orig all_models] = train_loo_cache(model, D, cached_scores, num_iter, Nouter, rate, prob_thresh, C, neg_feats)
% refines an initial part exemplar with new examples

if(~isfield(model, 'do_loo'))
   do_loo = 1;
else
   do_loo = model.do_loo;
end

if(nargout<=2)
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


Dneg = D;
cached_neg = cached_scores;

if(~isfield(model, 'hard_local') || model.hard_local==0)
   Dneg(inds) = [];
   cached_neg(inds) = [];
end

N = length(Dneg);

blocksize = min(blocksize, N);
%Nouter = 10;

if(~exist('neg_feats', 'var') || isempty(neg_feats))
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
for i = 1:length(neg_feats)
   bookkeeping{i} = zeros(1, size(neg_feats{i},2));
end

%Nouter = 1; % No latent search at the moment!

[pos_feats0 pos_im_ind0 dist0] = get_poselet_examples(model, Dpos, cached_pos); %collect_training_ex(model, Dpos, cached_pos, 1);
% Fix the features...
for p = 1:length(pos_feats0)
   if(~isempty(pos_feats0{p}))
      pos_feats0{p} = [pos_feats0{p}; zeros(1, size(pos_feats0{p},2))];
   end
end


for outer_iter = 1:Nouter
   if(outer_iter==1)  % Select top K examples using distance measure 
      dists = cat(2, dist0{:});
      dists = dists(1, :);

      Npos0 = numel(dists); %size(pos_feats,2);
      % Prune positives to top K
      if(rate<=1) % If rate is <= 1, it gives a ratio of examples, otherwise it specifies a fixed number of examples
         Kn = ceil(Npos0*rate);
      else
         Kn = rate;
      end

      [a b] = sort(dists, 'ascend');

      dist_th = a(min(end, Kn));

      pos_feats1 = {};
      pos_im_ind1 = {};
      for p = 1:length(pos_feats0)
         if(~isempty(pos_feats0{p}))
            ok = dist0{p}(1, :)<=dist_th;
            pos_feats1{p} = pos_feats0{p}(:, ok);
            pos_im_ind1{p} = pos_im_ind0{p}(ok);
         end
      end
   
   else
      % Select top examples based on estimated probability of being correct
      % Positive scores
      pos_scores = w(1:end-1)'*cat(2, pos_feats0{:}) + w(end);


      %neg_inds = 1:length(D);
      %neg_inds(pos_inds) = [];
      %neg_inds = 1:min(numel(Dneg), numel(Dpos)); %neg_inds(1:min(numel(Dpos),end));

     
      % Don't care about examples past the last positive example
      %scores = sort(pos_scores, 'descend');
      %model.thresh = scores(max(find(~isinf(scores))))-0.2; % Make it a little bit smaller so the last score bin corresponds to negatives

      %[dk neg_hyp] = collect_training_ex(model, Dneg(neg_inds), cached_neg(neg_inds), -1);
      %neg_hyp = cat(1, neg_hyp{:});
      %neg_scores = unique([neg_hyp.final_score]);

      % roc = computeROC([pos_scores(:); neg_scores(:)],  [ones(numel(pos_scores),1); -ones(numel(neg_scores), 1)]);
      %sig_param = learn_obj_prob(roc, 2);
      %sig_param = trainLogReg([pos_scores(:); neg_scores(:)]', [ones(numel(pos_scores),1); -ones(numel(neg_scores), 1)]', 0.00000001);
      %sig_param = trainLogReg([pos_scores(:); neg_scores(:)]', [ones(numel(pos_scores),1); -ones(numel(neg_scores), 1)]', 0.00000001);
      %pos_probs = sigmoid(pos_scores, sig_param); %score2prec(roc, feat_all(:, end));


      %ok = pos_probs>=prob_thresh;
      ok = pos_scores>=-1;
      score_thresh = min(pos_scores(ok));
   
      pos_feats1 = {};
      pos_im_ind1 = {};
      for p = 1:length(pos_feats0)
         if(~isempty(pos_feats0{p}))
            ok = w(1:end-1)'*pos_feats0{p} + w(end) >= score_thresh; %dist0{p}(1, :)<=dist_th;
            pos_feats1{p} = [pos_feats0{p}(:, ok)];
            pos_im_ind1{p} = pos_im_ind0{p}(ok);
         end
      end
   end

   pos_feats = cat(2, pos_feats1{:});
   pos_im_inds = cat(1, pos_im_ind1{:});

   Npos0 = size(pos_feats,2);
   Kn = Npos0; 

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
      model.thresh = -1; % 1/(1+exp(7)) is really small
      new_neg = collect_training_ex(model, Dneg(block), cached_neg(block), -1);

      [neg_feats bookkeeping] = update_neg_cache(neg_feats, bookkeeping, new_neg, block, w);

      % Update model
      feats = [pos_feats, cat(2, neg_feats{:})];
      labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];

      if(NO_SCORE_FEAT)
         feats(end,:) = 0; % Don't use the score of previous iterations
      end
      
      fprintf('Iter: %d/%d  - %d/%d\n', outer_iter, Nouter, inner_iter, ceil(N/blocksize));
      w0 = get_model_weights(model);

      Cf = C/Kn;
      pred = w0(1:end-1)'*pos_feats;

      %[a b] = sort(pred, 'descend');
      delta0 = ones(length(labels), 1);
      %delta0(1:size(pos_feats,2)) = 0;
      %delta0(b(1:min(end,Kn))) = 1; % Choose 

      [w deltas alphas obj] = subset_svm(feats, labels, Cf, Kn, delta0);

      %[w deltas alphas obj] = subset_logreg(feats, labels, Cf, ceil(sum(labels==1)*rate), delta0);
      delta_pos = deltas(1:size(pos_feats,2));
      clear feats; 
      all_models{end+1} = model;
      model = update_model_weights(model, w);
   end
end

% Compute which image each positive feature came from
%for i = 1:length(pos_hyp)
%   pos_im_ind{i} = repmat(i, length(pos_hyp{i}),1);
%end

%pos_im_inds = cat(1, pos_im_ind{:});

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
   % TODO: Put a parfor here! - Unfortunately the feature matrix could use lots of memory!
   for i = 1:length(pos_im_todo)
      fprintf('Computing leave one out estimate: %d/%d\n', i, length(pos_im_todo));
      curim = pos_im_todo(i);
   
      delta1 = [delta_pos(pos_im_inds~=curim); delta_neg];
      alpha0 = [alpha_pos(pos_im_inds~=curim); alpha_neg];
   
      pos_feats = cat(2, pos_feats1{[1:curim-1, curim+1:end]});
      feats = [pos_feats, cat(2, neg_feats{:})];
      if(NO_SCORE_FEAT)
         feats(end,:) = 0; % Don't use the score of previous iterations
      end
      
      labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];
   
      if(fast_subset) % Just train on precomputed subset
         %w_loo0{i} = svm_dual_mex(labels(delta1==1), feats(:, delta1==1), C)';
         if(numel(Cf)==1)
            w_loo0{i} = svm_dual_mex(labels(delta1==1), feats(:, delta1==1), Cf, [], alpha0)'; % Initialize using previous alpha
         else
            [wt b] = fast_svm(labels(delta1==1), feats(:, delta1==1), Cf(1), Cf(2)/Cf(1));
            w_loo0{i} = [wt(:); b];
         end
      else
         w_loo0{i} = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*rate));
      end
   %   [w_loo{curim} deltas_loo{curim} alphas_loo{curim}] = subset_svm(feats, labels, 1e-2, 288*rate);
      
   end

   w_loo = cell(1, length(pos_feats1));
   w_loo(pos_im_todo) = w_loo0;
   for i = 1:length(pos_feats1)
      if(~ismember(i, pos_im_todo))
         w_loo{i} = w;
      end
   end
else % don't do loo, just copy over model
   w_loo = cell(1, length(pos_feats1));
   for i = 1:length(pos_feats1)
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
   %neg_bookkeep{i}(scores{i}>=-7) = 0; % Logreg margin is -7 for now
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

