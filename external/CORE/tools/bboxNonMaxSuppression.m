function [keep2, group] = bboxNonMaxSuppression(bbox, score, maxoverlap)
% [keepind, group] = bboxNonMaxSuppression(bbox, score, minconf, maxov)
% bbox([x1 y1 x2 y2])

%ind = find([det(:).origconf]<minConf);
% keepind = find(score>=minconf);
% bbox2 = bbox(keepind, :);
% score2 = score(keepind, :);



    
% sort by confidence, lowest first
%[tmp, si] = sort([fdet(:).origconf]);
[tmp, si] = sort(score, 'descend');            
    
bbox = bbox(si, :);

tmpind = (1:numel(score));
keep = true(numel(score), 1); 

if nargout>1
    group = zeros(numel(score), 1, 'uint32');
end
ngrps = 0;
for i = 1:numel(score)             
    
    ov = getBoxOverlap(bbox(i, [1 3 2 4]), bbox(keep(1:i-1), [1 3 2 4]));
    
    if any(ov>maxoverlap)
        keep(i) = false;
        if nargout>1
            [maxov, ind] = max(ov);
            ind2 = tmpind(keep(1:i-1));
            group(i) = group(ind2(ind));
        end
    elseif nargout>1
        ngrps = ngrps+1;
        group(i) = ngrps;
    end
end
keep2 = false(size(score));
keep2(si(keep)) = true;

