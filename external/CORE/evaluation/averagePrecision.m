function [ap, appt] = averagePrecision(pr, r)
% [ap, appt] = averagePrecision(pr, r)

ap = zeros(size(pr));
appt = zeros(numel(pr), numel(r));
for k = 1:numel(pr)
    for k2 = 1:numel(r)
        ind = (pr(k).r>=r(k2));
        if any(ind)
          appt(k, k2) = max(pr(k).p(ind));          
        else
          appt(k, k2) = 0;
        end
        ap(k) = ap(k) + appt(k,k2)/numel(r);
    end
end