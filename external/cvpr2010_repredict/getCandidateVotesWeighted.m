function cand = getCandidateVotesWeighted(detections, pred, thresh)


cand.bbox = {};
cand.w = {};
cand.bbox_id = {};
cand.prednum = {};

% read in detections
 
for k = 1:numel(detections)
   bboxes = detections{k};
   
   if(isempty(bboxes))
      continue
   end

   ind = bboxes(:, end) > thresh;
   bboxes = bboxes(ind, :);
   
   if(isempty(bboxes))
      continue
   end

   keep = bboxNonMaxSuppression(bboxes(:, 1:4), bboxes(:, end), 0.5);

   bbox = bboxes(keep, :);
   cand.bbox_src{k} = bbox;
   h = bbox(:, 4)-bbox(:,2);  w = bbox(:,3)-bbox(:,1);    
   cx = (bbox(:, 3)+bbox(:,1))/2;  cy = (bbox(:,4)+bbox(:,2))/2;

   % predict object candidates from detections    
   dx = pred(k).offset2(:, 1); dy = pred(k).offset2(:, 2);
   dsx = pred(k).offset2(:, 3);  dsy = pred(k).offset2(:, 4);
   for k2 = 1:size(bbox, 1);
     sx = dsx*w(k2);  sy = dsy*h(k2);
     x = cx(k2) + dx*w(k2); y = cy(k2) + dy*h(k2);
     cand.bbox{end+1} = [x-sx/2 y-sy/2 x+sx/2 y+sy/2];
     cand.w{end+1} = pred(k).w2*bbox(k2, end);      
     cand.bbox_id{end+1} = ones(size(dx))*k2;
     cand.prednum{end+1} = ones(size(dx))*k;
   end
end
 
cand.bbox = cat(1, cand.bbox{:});
if isempty(cand.bbox)
   cand.bbox = zeros(0, 4);
end
cand.w = cat(1, cand.w{:});
cand.bbox_id = cat(1, cand.bbox_id{:});
cand.prednum = cat(1, cand.prednum{:});    
