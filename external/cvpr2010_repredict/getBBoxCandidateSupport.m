function [w, ids, w_ids] = getBBoxCandidateSupport(bbox, cand, ovthresh)
%  [w, ids] = getBBoxCandidateSupport(bbox, cand, ovthresh)
%  bbox should be sorted from highest score to lowest score

w =zeros(size(bbox,1),1);
ids = cell(size(bbox, 1), 1);
w_ids = cell(size(bbox, 1), 1);

if ~any(bbox)
    return;
end
    

ov = bbox_overlap_mex(bbox, cand.bbox);
%ov = ov';
ov(ov<ovthresh) = 0;
ov = ov ./ repmat(sum(ov, 1)+eps, size(ov,1), 1);

for k = 1:size(bbox, 1)    
    %ov = getBoxOverlap(bbox(k, [1 3 2 4]), cand.bbox(:, [1 3 2 4]));
    ids{k} = find(ov(k, :)>0); %=ovthresh);            
    w_ids{k} = cand.w(ids{k});
    w(k) = sum(w_ids{k}); %.*ov(k, :)'); %(ov>=ovthresh));
    ov(:, ids{k}) = 0;
end
