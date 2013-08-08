function roc = analyzeResult(result)

scores = cat(1, result.scores);
labels = cat(1, result.labels);

roc = computeROC(scores, labels); 
roc.npos = sum([result.npos]);  
roc.nim = numel(result); 
roc.nfound = sum([result.nfound]); 
roc.ndontcare = sum([result.ndontcare]);
roc.r = roc.r *roc.tp(end)/ roc.npos;
%roc.ap = averagePrecision(roc, (0:0.1:1));
roc.normAP = normalizedAP(roc, roc.npos, 0.15*roc.nim);
roc.ap = roc.normAP.ap;
[sv, si] = sort(scores, 'descend'); 
roc.labels = labels(si);
Nfp = 2;

roc.unfamiliar = cat(1, result.isholdout) & (labels==1);
roc.unfamiliar = roc.unfamiliar(si);
roc.familiar = ~roc.unfamiliar & (roc.labels==1); 
roc.nunfamiliar = sum([result.nholdout]);
roc.nfamiliar = roc.npos-roc.nunfamiliar;
roc.tp_familiar = cumsum(roc.familiar);
roc.tp_unfamiliar = cumsum(roc.unfamiliar);
roc.pct_found = roc.tp(end)/ roc.npos;
roc.area_2fp = computeROCAreaThresholded(roc, Nfp*numel(result))*roc.pct_found;

roc.pct_found_familiar = roc.tp_familiar(end)/ roc.nfamiliar;
roc.r_familiar = roc.tp_familiar/roc.tp_familiar(end)*roc.pct_found_familiar;
tmp.r = roc.r_familiar;
tmp.tp = roc.tp_familiar;
tmp.fp = roc.fp;
tmp.p = roc.tp_familiar ./ (roc.tp_familiar + roc.fp);
tmp.conf = roc.conf;
%roc.ap_familiar = averagePrecision(tmp, (0:0.1:1));
roc.normAP_familiar = normalizedAP(tmp, roc.nfamiliar, 0.15*roc.nim);
roc.ap_familiar = roc.normAP_familiar.ap;

roc.area_familiar_2fp = computeROCAreaThresholded(tmp, Nfp*numel(result))*roc.pct_found_familiar;
roc.pct_found_unfamiliar = roc.tp_unfamiliar(end)/ roc.nunfamiliar;
roc.r_unfamiliar = roc.tp_unfamiliar/roc.tp_unfamiliar(end)*roc.pct_found_unfamiliar;
tmp.r = roc.r_unfamiliar;
tmp.tp = roc.tp_unfamiliar;
tmp.p = roc.tp_unfamiliar ./ (roc.tp_unfamiliar + roc.fp);
%roc.ap_unfamiliar = averagePrecision(tmp, (0:0.1:1));
roc.normAP_unfamiliar = normalizedAP(tmp, roc.nunfamiliar, 0.15*roc.nim);
roc.ap_unfamiliar = roc.normAP_unfamiliar.ap;

roc.area_unfamiliar_2fp = computeROCAreaThresholded(tmp, Nfp*numel(result))*roc.pct_found_unfamiliar;

roc = orderfields(roc);
fprintf('AP:             overall=%.3f   familiar=%.3f   unfamiliar=%.3f\n', roc.ap, roc.ap_familiar, roc.ap_unfamiliar);
fprintf('Norm AP:        overall=%.3f   familiar=%.3f   unfamiliar=%.3f\n', roc.normAP.apn, roc.normAP_familiar.apn, roc.normAP_unfamiliar.apn);
fprintf('Area at 2 FP:   overall=%.3f   familiar=%.3f   unfamiliar=%.3f\n', roc.area_2fp, roc.area_familiar_2fp, roc.area_unfamiliar_2fp);
fprintf('Counts: pos=%d    found=%d     dontcare=%d     familiar=%d   unfamiliar=%d\n', roc.npos, roc.nfound, roc.ndontcare, roc.nfamiliar, roc.nunfamiliar);

if(1)
   subplot(1,2,1), hold off, plot(roc.fp/roc.nim, roc.tp/roc.npos, 'r', 'linewidth', 1);
   hold on, plot(roc.fp/roc.nim, roc.tp_familiar/roc.nfamiliar, '--g', 'linewidth', 1);
   hold on, plot(roc.fp/roc.nim, roc.tp_unfamiliar/roc.nunfamiliar, '--b', 'linewidth', 1);
   axis([0 5 0 1])
   subplot(1,2,2), hold off, plot(roc.r, roc.tp./(roc.tp+roc.fp) , 'r', 'linewidth', 1);
   hold on, plot(roc.r_familiar,roc.tp_familiar./(roc.tp_familiar+roc.fp), '--g', 'linewidth', 1);
   hold on, plot(roc.r_unfamiliar,roc.tp_unfamiliar./(roc.tp_unfamiliar+roc.fp), '--b', 'linewidth', 1);        
   axis([0 1 0 1])
   drawnow;
end
