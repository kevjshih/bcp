function [w delta alphas obj mw] = subset_svm(x, labels, weights, C, Kn, delta0, reg)
% w = subset_svm(x, labels, C, K)
% x = Nf x Nex
% labels = Nex
% C - regularization
% K - 
Nex = size(x,2);
% add bias
%x = [x; ones(1, Nex)];
Nf = size(x,1);


if(numel(labels)~=Nex)
   error('Number of labels doesn''t match size of examples');
end

labels = reshape(labels, Nex, 1);


alphas = [];
% N approaches:
% a) Linear constraints
% b) Coarse iterative procedure
% c) Tight iterative procedure

% Linear constraints
   w0 = zeros(Nf, 1);
   delta0 = ones(Nex, 1);
   delta_prev = zeros(Nex,1);
   i=0;

   pos = find(labels==1);

   while sum(delta0~=delta_prev)>1 % >1 avoids oscillations, it's close enough
      i = i + 1;
      fprintf('====================%d=================\n', i);
      if(exist('reg','var'))
         [w0 alphas] = svm_weighted_dual_mex(labels(delta0==1), x(:, delta0==1), weights(delta0==1), C, reg);
      else
         [w0 alphas] = svm_weighted_dual_mex(labels(delta0==1), x(:, delta0==1), weights(delta0==1), C);
      end
      
      w0 = w0';
      fprintf('Support vectors: %d(pos:%d, neg:%d)\n', sum(alphas>1e-5), sum(alphas(labels(delta0==1)==1)>1e-5), sum(alphas(labels(delta0==1)==-1)>1e-6));


   delta_prev = delta0;
   % Minimize loss
   pred = weights(pos)'.*(1 - (w0(1:end-1)'*x(:, pos) + w0(end))); 
 
   if(Kn<1) 
      p = tic;
      A = reshape(weights(pos), 1, [])/sum(weights(pos));
      delta0_pos = full_knapsack(pred, A, Kn);
      toc(p);
      delta0(pos) = delta0_pos;
   else
      [a b] = sort(pred, 'ascend');
      delta0(pos) = 0;
      delta0(pos(b(1:min(end,Kn)))) = 1; % Choose the least violated examples
   end
   %   delta0(pos(pred>=1)) = 1; % Include all examples that aren't violated

   %delta0 = ((1-labels'.*pred)<1/K)';
   fprintf('Iter: %d. #Changed: %f Ignored Examples: Pos:%d/%d, Neg:%d/%d\n', i, sum(delta0~=delta_prev)/2, sum(labels(~delta0)==1), sum(labels==1), sum(labels(~delta0)==-1), sum(labels==-1));

   end

w = w0;
delta = delta0;
