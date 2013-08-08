function result = averagePrecisionNormalized(varargin)
%Case 1: ...(conf, label, npos, nnorm)
%Case 2: ...(roc.{fp,tp,r,p,conf}, npos, nnorm)
% result = averagePrecisionNormalized(conf, label, npos, nnorm)
%
% Computes full interpolated average precision, first normalizing the 
% precision values.  
% Normally, p = tp ./ (fp + tp), but this is sensitive to the density of
% positive examples.  For normalized precision, 
%   tp2 = (tp*N/Npos);  p_norm = tp2 ./ (fp + tp2);
%
% Input:
%   conf(ndet, 1): confidence of each detection
%   label(ndet, 1): label of each detection (-1, 1; 0=don't care)
%   npos: the number of ground truth positive examples
%   nnorm: the normalized value for number of positives (for normalized AP)
%
% Output:
%   result.(labels, conf, npos, nnorm, r, p, pn, ap, apn, ap_std, apn_std):
%     the precision-recall and normalized precision-recall curves with AP
%     and standard error of AP

if(~isstruct(varargin{1}))
   conf = varargin{1};
   label = varargin{2};
   npos = varargin{3};
   nnorm = varargin{4};

   [sv, si] = sort(conf, 'descend');
   label = label(si);
   
   tp = cumsum(label==1);
   fp = cumsum(label==-1);
   conf = sv;
else
   roc = varargin{1};
   npos = varargin{2};
   nnorm = varargin{3};

   tp = roc.tp;
   fp = roc.fp;
   conf = roc.conf;
end
r = tp / npos;
p = tp ./ (tp + fp);

tpn = tp*nnorm/npos;
pn = tpn ./ (tpn + fp);

%result = struct('labels', label, 'conf', conf, 'r', r, 'p', p, 'pn', pn);
result.npos = npos;
result.nnorm = nnorm;


% compute interpolated precision and normalized precision
%istp = (label==1);
istp = [tp(1)>0; tp(2:end)~=tp(1:end-1)];

Np = numel(r);
for i = Np-1:-1:1
  p(i) = max(p(i), p(i+1));
  pn(i) = max(pn(i), pn(i+1));
end
result.ap = mean(p(istp))*r(end);
result.apn = mean(pn(istp))*r(end);

missed = zeros(npos-max(tp),1);
result.ap_stderr = std([p(istp(:)) ; missed])/sqrt(npos);
result.apn_stderr = std([pn(istp(:)) ; missed])/sqrt(npos);
