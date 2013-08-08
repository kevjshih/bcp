function [w_out acc_best obj]= learnLogRegw(x_in, y_in, lambda, weighting, val, w0)
% Train weighted binary logistic regression

%n_ex = size(x_in,1);
x = x_in; %, ones(n_ex, 1)]; % Add the bias ...
y = y_in(:);

n_classes = 2;
n_feats = size(x,1);

%w0 = reshape(zeros(n_feats, n_classes),[],1);
if(~exist('w0', 'var') || isempty(w0))
   w0 = zeros(n_feats+1,1);
else
   w0 = w0(:);
end

options = optimset('Display','off');
%options = optimset(options, 'DerivativeCheck','on');
%options = optimset(options, 'NumDiff',1);
%options.NumDiff = 1;

if(~exist('weighting', 'var') || isempty(weighting))
   weighting = ones(numel(y_in),1)/numel(y_in);
end

% Normalize so everything sums to 1
weighting(1:end) = 1/length(weighting); %(y_in==1) = weighting(y_in==1)*2/sum(weighting(y_in==1));

% Do cross validation for best parameter
if(exist('val','var') && ~isempty(val) && val>0)
   r = 1:length(y_in);%randperm(length(y_in));
   tr_i = r(1:2:end);%ceil(val*end));
   te_i = r(2:2:end);%ceil(val*end)+1:end);

   for i = 1:length(lambda)
      wt = minFunc(@(w)logregMC(w, x(:, tr_i), y(tr_i), lambda(i), weighting(tr_i)), w0, options);
      w{i} = wt; %weighting.*reshape(wt, n_feats, n_classes);
      scores = wt(1:end-1)'*x(:, te_i)+wt(end); %predLogRegMCw(x(te_i,:), w{i},0);
 
      rocSt = computeROC(scores(:), y(te_i));
      roc(i) = ap(rocSt.r, rocSt.p)%[rocSt.area])%sum([rocSt.area]'.*class_dist)/sum(class_dist);
      %roc(i) = mean([rocSt.area])%sum([rocSt.area]'.*class_dist)/sum(class_dist);
   end

   %[acc_best best_lambda] = max(acc);
   [acc_best best_lambda] = max(roc);
   fprintf('Best accuracy: %f (lambda=%f)\n', acc_best, lambda(best_lambda));
else
   best_lambda = 1;
   acc_best = [];
end

[w obj] = minFunc(@(w)logregMC(w, x, y, lambda(best_lambda), weighting), w0, options);
fprintf('Minimum value: %f\n', obj);
w_out = w;

function [f g] = logregMC(w0, x, y, lambda, weighting)

w = w0(:);

pred = [w(1:end-1)'*x + w(end)]';

pred_by_cl = exp(-y.*pred);

denom = 1 + pred_by_cl;

f = lambda/2*(sum(w(1:end-1).^2) + w(end)*0.001) + weighting'*(log(1 + pred_by_cl));

g = lambda*[w(1:end-1); w(end)*0.001] - [x*(weighting.*y.*pred_by_cl./denom); weighting'*(y.*pred_by_cl./denom)];
