function [w delta alphas obj mw] = subset_svm(x, labels, C, Kn, delta0, reg)
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
if(0)
   opts.method = 'lbfgs';
   opts.maxIter = 2000;
   opts.verbose = 2;
   %opts.optTol = 1e-;
   lb = [-inf(Nf+1, 1); zeros(Nex, 1)];
   ub = [inf(Nf+1, 1); ones(Nex, 1)];
   w0 = rand(Nf+1, 1);
   delta0 = ones(Nex, 1);

%w0 = rand(numel(w0)+1, 1);
%delta0 = rand(size(delta0))>0.5;


   for i = 1:10
   %w_all = minConf_TMP(@(w)obj_w_delta(w, x, labels, C, K), [w0; delta0], lb, ub,opts);

      % Fmincon
      opts = optimset('gradobj', 'on', 'display', 'iter', 'algorithm', 'sqp');
      w_all = fmincon(@(w)obj_w_delta(w, x, labels, C, K), [w0; delta0], [],[],[],[],lb, ub,[],opts);

      w0 = w_all(1:Nf+1);
      delta0 = double(w_all(Nf+2:end)>0.75);
   end


   w = w0;
   delta = delta0;

elseif(1) % Iterative
   opts.method = 'lbfgs';
   opts.maxIter = 10000; % As many iterations as you need
   opts.verbose = 2;
   lb = [-inf(Nf, 1)];
   ub = [inf(Nf, 1)];
   w0 = zeros(Nf, 1);

   if(~exist('delta0', 'var'))
      delta0 = ones(Nex, 1);
   end

   delta_prev = zeros(Nex,1);
   i=0;

   pos = find(labels==1);
%   r = randperm(length(pos));
%   delta0(pos(r(Kn+1:end))) = 0;

   while any(delta0~=delta_prev)
      i = i + 1;
%for i = 1:50
      fprintf('====================%d=================\n', i);
%   [w_all f(i)] = minConf_TMP(@(w)obj_w(w, delta0, x, labels, C, K), w0, lb, ub,opts);

      switch 'fast'
         case 'linear'
            mw = lintrain(labels(delta0==1), sparse(x(:, delta0==1)'), sprintf('-s 3 -c %f', C));
            w0 = mw.w';
         case 'libsvm'
            disp(size(x));
            mw = libsvmtrain(labels(delta0==1), x(:, delta0==1)', sprintf('-s 0 -t 0 -c %f ', C));
            w0 = [mw.sv_coef'*mw.SVs, -mw.rho]';
         case 'sgd'
            w0 = svm_sgd(labels(delta0==1), x(:, delta0==1), C)';
         case 'fast'
            if(exist('reg','var'))
               [w0 alphas obj] = svm_dual_mex(labels(delta0==1), x(:, delta0==1), C, reg);
            else
               [w0 alphas obj] = svm_dual_mex(labels(delta0==1), x(:, delta0==1), C);
            end
            w0 = w0';
            fprintf('Support vectors: %d(pos:%d, neg:%d)\n', sum(alphas>1e-5), sum(alphas(labels(delta0==1)==1)>1e-5), sum(alphas(labels(delta0==1)==-1)>1e-5));
      end


      delta_prev = delta0;
      pred = w0(1:end-1)'*x(:, pos) + w0(end); 
      
      [a b] = sort(pred, 'descend');
      delta0(pos) = 0;
      delta0(pos(b(1:min(end,Kn)))) = 1; % Choose the least violated examples
%      delta0(pos(pred>=1)) = 1; % Include all examples that aren't violated

      %delta0 = ((1-labels'.*pred)<1/K)';
      fprintf('Iter: %d. #Changed: %d Ignored Examples: Pos:%d/%d, Neg:%d/%d\n', i, sum(delta0~=delta_prev)/2, sum(labels(~delta0)==1), sum(labels==1), sum(labels(~delta0)==-1), sum(labels==-1));
   end
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
