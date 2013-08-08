function roc = expand_roc_sm(roc_sm, w)

if(~exist('w', 'var'))
   w = 1;
end

roc.conf = roc_sm.conf; %sc_sort(:);
roc.tp = cumsum(full(roc_sm.tp));
roc.fp = cumsum(full(~roc_sm.tp));
roc.fp_nodup = cumsum(full(~roc_sm.tp) & full(~roc_sm.dup));

roc.p = roc.tp ./ (roc.tp + w*roc.fp);
roc.r = roc.tp / roc_sm.Npos;
roc.ap = VOCap(roc.r, roc.p);

roc.p_nodup = roc.tp ./ (roc.tp + w*roc.fp_nodup);
roc.ap_nodup = VOCap(roc.r, roc.p_nodup);
roc.Npos = roc_sm.Npos;
