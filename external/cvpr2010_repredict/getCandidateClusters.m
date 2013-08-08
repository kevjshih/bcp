function [bbox2, w2, assignment] = getCandidateClusters(bbox, thresh, w, tol, maxc)
% [bbox2, w2] = getCandidateClusters(bbox, thresh, w, tol)
%
% Gets cluster centers of bounding box based on intersection over union distance.  
% bbox(nbox, :) = [x1 y1 x2 y2]
% Thresh provides overlap threshold for determining number of clusters.  
% w is weight over bbox candidates
 

nbox = size(bbox, 1);
weighted = true;

if ~exist('w', 'var') || isempty(w)
  w = ones(nbox, 1);
  weighted = false;
end
if ~exist('tol', 'var') || isempty(tol)
  tol = 0;
end
if ~exist('maxc', 'var') || isempty(maxc)
  maxc = Inf;
end

[sv, si] = sort(w, 'descend');
bbox = bbox(si, :);
w = w(si);

assignment = zeros(nbox, 1);
unassigned = 1:nbox;

bbox2 = zeros(nbox, 4);
nc = 0;

nums = 1:nbox;
while ~isempty(unassigned)

  if ~weighted
    k = unassigned(ceil(rand(1)*numel(unassigned))); % Randomly select unassigned box
  else
    [mv, mi] = max(w(unassigned)); % Select highest scoring unassigned box
    k = unassigned(mi);
  end

  nc = nc+1;
  bbox2(nc, :) = bbox(k, :);
  
  ov = bbox_overlap_mex(bbox(k, :), bbox(unassigned, :));
  matched = ov>=thresh;
  if ~any(matched)
    keyboard;
  end
  
  assignment(unassigned(matched)) = nc;
  unassigned = unassigned(~matched);

  if nc==maxc      
      break;
  end
  
%   if mod(nc, 1000)==0
%     disp([num2str(nc) '   ' num2str(numel(unassigned))])
%   end
end


    

bbox2 = bbox2(1:nc, :);
w2 = zeros(nc, 1);

change = Inf;
changedc = true(nc, 1);
ov = zeros(nbox, nc, 'single');
oldov = ones(nbox, nc, 'single');
changedb = zeros(nc, 1);

while change > tol
  oldassign = assignment;  

  % compute means
  for k = 1:nc
    if changedc(k)
      ind = (assignment==k);
      w2(k) = sum(w(ind));
      if w2(k)==0
         keyboard;
      end
      oldbb = bbox2(k, :);
      
      % Weighted average of boxes:
      bbox2(k, :) = sum(bbox(ind, :).*repmat(w(ind), [1 size(bbox2, 2)]), 1) ./ w2(k);
      
      changedb(k) = (1-bbox_overlap_mex(oldbb, bbox2(k, :)));
    else
      changedb(k) = 0;
    end
  end
  
  % recompute assignments
  for k = 1:nc
    if changedc(k)
        ind = oldov(:, k)>thresh/2;
        ov(ind, k) = bbox_overlap_mex(bbox2(k, :), bbox(ind, :));
    end
  end
  [mv, assignment] = max(ov, [], 2);
  assignment(mv<thresh/2) = 0;
  
  for k = 1:numel(assignment) 
    if assignment(k)~=oldassign(k)  
      if assignment(k)>0
        changedc(assignment(k)) = true;
      end
      if oldassign(k)>0
        changedc(oldassign(k)) = true;
      end
    end
  end 
  oldov = ov;
  change = sum(changedb.*w2) / sum(w2); % average change in bounding box, weighted by point

  %disp(num2str([change change2]))
end 

