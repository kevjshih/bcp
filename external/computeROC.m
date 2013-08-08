function roc = computeROC(scores, labels, w)
% From Derek
if(nargin<3)
    w = 1;
end

[sval, sind] = sort(scores, 'descend');
Npos = sum(labels==1);

ignore = zeros(size(isinf(sval)));

roc.tp = cumsum(labels(sind(~ignore))==1);
roc.fp = cumsum(labels(sind(~ignore))==-1);
roc.conf = sval(~ignore);

roc = computeROCArea(roc);

% ind = find([true (roc.conf(2:end)~=roc.conf(1:end-1))']);
% dfp = roc.fp(ind(2:end)) - roc.fp(ind(1:end-1));
% avetp = (roc.tp(ind(1:end-1))+roc.tp(ind(2:end)))/2;
% roc.area = sum(avetp/roc.tp(end) .* dfp/roc.fp(end));

roc.p = roc.tp ./ (roc.tp + w*roc.fp);
roc.r = roc.tp / Npos;

roc.p(isinf(sval) & sval<0) = 0;