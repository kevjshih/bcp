function [model w_loo w_orig all_models neg_feats] = train_boosted(model, D, cached_scores, num_iter, Nouter, rate, C, part_ind)

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

Dneg = D;
cached_neg = cached_scores;
Dneg(inds) = [];
cached_neg(inds) = [];



[model.part(part_ind).computed] = deal(0);
keyboard
% Compute thresh so that ~100k top scoring negative examples are mined
topscores = [];
numtokeep = 1e5;
for i = 1:length(cached_neg)
   if(~isempty(cached_neg{i}.part_scores))
      t = unique(cached_neg{i}.part_scores(:, part_ind));
      [topscores b] = sort([topscores; t(end:-1:1)], 'descend');
      topscores = topscores(1:min(numtokeep,end));
   end
end

model.thresh = topscores(end);

if(isfield(model, 'hard_local') && model.hard_local==1)
   Dloc = D(inds);
   cached_loc = cached_scores(inds);
   [loc_feats0 loc_hyp] = collect_negatives(model, Dloc, cached_loc);
end

model.nms = 0.7;

[pos_feats0 pos_hyp] = collect_positives(model, Dpos, cached_pos);
[neg_feats0 neg_hyp] = collect_negatives(model, Dneg, cached_neg); % Think about how to speed this up

% Now what????????
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

pos_feats = cat(2, pos_feats0{:});
Npos = size(pos_feats,2);
%Cf = C/Npos;


[neg_feats bookkeeping] = update_neg_cache(neg_feats, bookkeeping, new_neg, block, w); % Remove any duplicates!

% Update model
feats = [pos_feats, cat(2, neg_feats{:})];
labels = [ones(size(pos_feats,2), 1); -ones(size(feats,2)-size(pos_feats,2),1)];

if(NO_SCORE_FEAT)
    feats(end,:) = 0; % Don't use the score of previous iterations
end
      
w0 = get_model_weights(model);



% Train it!!
Kn = ceil(sum(labels==1)*rate);
Cf = C/Kn;
pred = w0(1:end-1)'*pos_feats;

[a b] = sort(pred, 'descend');
delta0 = ones(length(labels), 1);
delta0(1:size(pos_feats,2)) = 0;
delta0(b(1:min(end,Kn))) = 1; % Choose 

[w deltas alphas obj] = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*rate), delta0);
delta_pos = deltas(1:size(pos_feats,2));
model = update_model_weights(model, w);



%%%%%%%%%%%%%%%%%% Compute which image each positive feature came from
% LEAVE ONE OUT
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

