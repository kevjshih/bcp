function ov = bbox_overlap(bbox, bboxes, quick_th)

if(isempty(bboxes))
    ov = [];
    return;
end

if(exist('quick_th', 'var') && quick_th>0) % Assumes every box has the same size
   box_area = (bboxes(1,3)-bboxes(1,1)+1)*(bboxes(1,4)-bboxes(1,2)+1);
   reg_area = (bbox(:,3)-bbox(:,1)+1).*(bbox(:,4)-bbox(:,2)+1);

   ratio = reg_area/box_area;
   todo = reshape(find(ratio>quick_th & ratio<1/quick_th), 1, []);
else
   todo = 1:size(bbox,1);
end
slow = 0;
if(slow)
ov = zeros(size(bboxes, 1), size(bbox,1));

for i = todo
   min_x = bboxes(:,1);
   min_y = bboxes(:,2);
   max_x = bboxes(:,3);
   max_y = bboxes(:,4);

   bi = [max(min_x, bbox(i, 1)), max(min_y, bbox(i, 2)), min(max_x, bbox(i, 3)), min(max_y, bbox(i, 4))];
   iw = bi(:,3) - bi(:,1) + 1;
   ih = bi(:,4) - bi(:,2) + 1;

   ua = (max_x - min_x + 1).*(max_y - min_y + 1) + (bbox(i,3) - bbox(i,1) + 1).*(bbox(i,4)-bbox(i,2)+1) - iw.*ih;
   
   ov(:, i) = iw.*ih./ua;
   ov(iw<=0 | ih<=0, i) = 0; %No overlap
end

else
bbox2 = bbox(todo,:)';
box_area = (bbox2(3,:)-bbox2(1,:)+1).*(bbox2(4,:)-bbox2(2,:)+1);
bboxes_area = (bboxes(:,3)-bboxes(:,1)+1).*(bboxes(:,4)-bboxes(:,2)+1);

iw = bsxfun(@min, bboxes(:,3), bbox2(3,:)) - bsxfun(@max, bboxes(:,1), bbox2(1,:)) + 1;
ih = bsxfun(@min, bboxes(:,4), bbox2(4,:)) - bsxfun(@max, bboxes(:,2), bbox2(2,:)) + 1;
intersect = iw.*ih;
ua = bsxfun(@plus, bboxes_area, box_area) - intersect;

ov2 = intersect./ua;
ov2(iw<=0 | ih<=0) = 0;

ov = zeros(size(ov2,1), size(bbox,1));
ov(:,todo) = ov2;
end