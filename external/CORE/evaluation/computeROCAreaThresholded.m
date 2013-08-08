function area = computeROCAreaThresholded(roc, maxfp)
% area = computeROCAreaThresholded(roc, maxfp)

t = max(find(roc.fp == maxfp));
if isempty(t),
    t = max(find(roc.fp<=maxfp));
end
roc.fp = roc.fp(1:t);
pctfound = roc.tp(t)/roc.tp(end);
roc.tp = roc.tp(1:t);
roc.conf = roc.conf(1:t);

roc = computeROCArea(roc);
area = roc.area*pctfound;

if roc.fp(end)<maxfp
    area = area*roc.fp(end)/maxfp + (maxfp-roc.fp(end))/maxfp;
end