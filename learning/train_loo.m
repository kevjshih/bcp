function [model w_loo w_orig all_models neg_feats] = train_loo(model, ...
                                                     D, cached_scores, num_iter, Nouter, rate, C, neg_feats)
   % DEPRECATED: use train_loo_cache instead

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
   model.cached_weight = 0;
else
   model.cached_weight = 10; % Start by emphasizing previous results
end

if(~exist('C','var'))
   C = 1e-2;
end

for outer_iter = 1:Nouter
   [pos_feats0 pos_hyp] = collect_positives(model, Dpos, cached_pos);
   pos_feats = cat(2, pos_feats0{:});
   Npos = size(pos_feats,2);
   Cf = C/Npos;

   r = randperm(N);

   inner_iter = 0;
   obj_prev = -inf;
   obj = 0;

   %while(inner_iter<num_iter)% && abs(obj-obj_prev)/obj>0.005) <-- This only works when you loop over all images 
   for inner_iter = 1:num_iter%min(num_iter, ceil(N/blocksize))
      obj_prev = obj;

      %first = (inner_iter-1)*blocksize+1;
      %last = min(inner_iter*blocksize, N);
      block = sort(ceil(rand(blocksize,1)*N));%sort(r(first:last));

      neg_feats(block) = collect_negatives(model, Dneg(block), cached_neg(block));
     
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
         Kn = ceil(sum(labels==1)*rate);
         pred = w0(1:end-1)'*pos_feats;

         [a b] = sort(pred, 'descend');
         delta0 = ones(length(labels), 1);
         delta0(1:size(pos_feats,2)) = 0;
         delta0(b(1:min(end,Kn))) = 1; % Choose 


         [w deltas alphas obj] = subset_svm(feats, labels, Cf, ceil(sum(labels==1)*rate), delta0);
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
         w_loo0{i} = svm_dual_mex(labels(delta1==1), feats(:, delta1==1), Cf, [], alpha0)'; % Initialize using previous alpha
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

return
if(0)
% Compare deltas
real_deltas = -ones(length(pos_im_inds), length(pos_im_todo));

for i = 1:length(pos_im_todo)
   curim = pos_im_todo(i);
   notin = find(pos_im_inds~=curim);

   real_deltas(notin, i) = deltas_loo{curim}(1:length(notin));  
end

   if(~isempty(pos_feats0{i}));
      scores_loo{i} = w_loo{i}(1:end-1)'*pos_feats0{i} + w_loo{i}(end);
      scores_orig{i} = w(1:end-1)'*pos_feats0{i} + w(end);
   end
end
end
