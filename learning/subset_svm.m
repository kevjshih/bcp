function [w delta alphas obj mw] = subset_svm(x, labels, C, Kn, delta0, reg, split)
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
w0 = zeros(Nf, 1);

if(~exist('delta0', 'var'))
   delta0 = ones(Nex, 1);
end

delta_prev = zeros(Nex,1);
i=0;

pos = find(labels==1);

if(~exist('split', 'var'))
   split = 1;
end

partition = max(1, ceil(rand(size(delta0))*split));
                                 % If using multiple alternate splits, make sure it touches all splits several times
while any(delta0~=delta_prev) || (split>1 && i<2*split)
   i = i + 1;
   fprintf('====================%d=================\n', i);

   this_time = partition == (mod(i, split)+1);
   this_time_ind = find(this_time);

   if(exist('reg','var') && ~isempty(reg))
      [w0 alphas obj] = svm_dual_mex(labels(delta0==1 & this_time), x(:, delta0==1 & this_time), C, reg);
   elseif(numel(C)==2)
      weighting = ones(size(labels(delta0==1)));
      weighting(labels(delta0==1)==1) = C(2)/C(1);
      [w0 alphas obj] = svm_weighted_dual_mex(labels(delta0==1 & this_time), x(:, delta0==1 & this_time), weighting, C(1));
   else
      [w0 alphas obj] = svm_dual_mex(labels(delta0==1 & this_time), x(:, delta0==1 & this_time), C);
   end
   w0 = w0';
   fprintf('Support vectors: %d(pos:%d, neg:%d)\n', sum(alphas>1e-5), sum(alphas(labels(delta0==1 & this_time)==1)>1e-5), sum(alphas(labels(delta0==1 & this_time)==-1)>1e-5));

   delta_prev = delta0;
    
   pos_this_time = intersect(pos, this_time_ind);
   pred = w0(1:end-1)'*x(:, pos_this_time) + w0(end); 
   
   [a b] = sort(pred, 'descend');
   delta0(pos_this_time) = 0;
   delta0(pos_this_time(b(1:min(end,Kn)))) = 1; % Choose the least violated examples

   fprintf('Iter: %d. #Changed: %d Ignored Examples: Pos:%d/%d, Neg:%d/%d\n', i, sum(delta0~=delta_prev)/2, sum(labels(~delta0)==1), sum(labels==1), sum(labels(~delta0)==-1), sum(labels==-1));
end

if(split>1) % Retrain with all of the examples....
   % Retraining!!
   fprintf('Retraining!!!!\n');
   if(exist('reg','var') && ~isempty(reg))
      [w0 alphas obj] = svm_dual_mex(labels(delta0==1), x(:, delta0==1), C, reg);
   elseif(numel(C)==2)
      weighting = ones(size(labels(delta0==1)));
      weighting(labels(delta0==1)==1) = C(2)/C(1);
      [w0 alphas obj] = svm_weighted_dual_mex(labels(delta0==1), x(:, delta0==1), weighting, C(1));
   else
      [w0 alphas obj] = svm_dual_mex(labels(delta0==1), x(:, delta0==1), C);
   end
   w0 = w0(:);
end

w = w0;
delta = delta0;

function [f g] = obj_w(w0, deltas, x, labels, C, K)
% w = Nf x 1
% deltas = Nex x 1
% x = Nf x Nex
% labels = Nex x 1

Nex = size(x,2);
Nf = size(x,1);

w = w0(1:Nf);
b = w0(Nf+1);

risk = max(0, 1 - labels'.*(w'*x + b)); % 1 x Nex

f = 1/2*w'*w + C*deltas'*risk' + C/K*sum(1-deltas);

if(nargout>=2)
   g_w = w - C*x*(labels.*deltas.*(risk>0)'); % Nf x 1
   g_b = C*sum(labels.*deltas.*(risk>0)');

   g = [g_w; g_b];
end

function [f g] = obj_w_delta(w0, x, labels, C, K)
% w0 = Nf+Nex x 1
% x = Nf x Nex
% labels = Nex x 1
Nex = size(x,2);
Nf = size(x,1);

w = w0(1:Nf);
b = w0(Nf+1);
deltas =w0(Nf+2:end);


risk = max(0, 1 - labels'.*(w'*x + b)); % 1 x Nex

f = 1/2*w'*w + C*deltas'*risk' + C/K*sum(1-deltas);

if(nargout>=2)
   fprintf('+');
   g_w = w - C*x*(labels.*deltas.*(risk>0)'); % Nf x 1
   g_b = -C*sum(labels.*deltas.*(risk>0)');
   g_delta = C*(risk' - 1/K); %C*(risk' - 1/K);
   g = [g_w; g_b; g_delta];
else
   fprintf('.');
end
