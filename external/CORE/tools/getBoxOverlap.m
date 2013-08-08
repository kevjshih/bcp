function ov = getBoxOverlap(box, boxes, border)
% ov = computeOverlap(box, boxes, include)
% Returns area(box AND boxes) / area(box OR boxes)
% box = [x1 x2 y1 y2]
% boxes(nboxes, :) = [x1 x2 y1 y2]
% border = 1 (default) if width of bounding box is x2-x1+1 (so that
%                       bounding box line is included in box)

if ~exist('border', 'var') || isempty(border)
    border = 1;
end

if size(box, 1)>1
    ov = zeros(size(boxes, 1), size(box, 1), 'single');
    for k = 1:size(box, 1)
        ov(:, k) = getBoxOverlap(box(k, :), boxes, border);        
    end
    return;
end    

nboxes = size(boxes, 1);
ov = zeros(nboxes, 1);

bi=[max(box(1),boxes(:, 1))  max(box(3),boxes(:, 3))  ...
    min(box(2),boxes(:, 2))  min(box(4),boxes(:, 4))];

iw=bi(:, 3)-bi(:, 1)+border;
ih=bi(:, 4)-bi(:, 2)+border;

ind = iw >0 & ih > 0; % others have no intersection
if any(ind)
    a1 = (boxes(ind, 2)-boxes(ind, 1)+border).*(boxes(ind, 4)-boxes(ind, 3)+border);
    a2 = (box(2)-box(1)+border)*(box(4)-box(3)+border);    

    intersectArea = iw(ind).*ih(ind);
    unionArea = a1 + a2 - intersectArea;
    ov(ind) = intersectArea ./ unionArea;
end


% for i = 1:size(boxes, 1)    
%     
%     boxgt = boxes(i, :);
%     bi=[max(box(1),boxgt(1)) ; max(box(3),boxgt(3)) ; min(box(2),boxgt(2)) ; min(box(4),boxgt(4))];
% 
%     iw=bi(3)-bi(1)+1;
%     ih=bi(4)-bi(2)+1;
% 
%     a1 = (boxgt(2)-boxgt(1)+1)*(boxgt(4)-boxgt(3)+1);
%     a2 = (box(2)-box(1)+1)*(box(4)-box(3)+1);    
%     
%     ov(i) = 0;
%     if iw>0 && ih>0                
%         ua = a1 + a2 - iw*ih;        
%         ov(i)=iw*ih/ua;
%     end        
%     
% end