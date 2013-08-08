function ov = bbox_overlap(bbox, bboxes, quick_th)

if(isempty(bboxes))
    ov = [];
    return;
end

min_x = bboxes(:,1);
min_y = bboxes(:,2);
max_x = bboxes(:,3);
max_y = bboxes(:,4);

bi = [max(bboxes(:, 1), bbox(:, 1)), max(bboxes(:, 2), bbox(:, 2)), min(bboxes(:,3), bbox(:,3)), min(bboxes(:, 4), bbox(:,4))];
iw = bi(:, 3) - bi(:, 1) + 1;
ih = bi(:, 4) - bi(:, 2) + 1;

ua = (max_x - min_x + 1).*(max_y - min_y + 1) + (bbox(:,3) - bbox(:,1) + 1).*(bbox(:,4)-bbox(:,2)+1) - iw.*ih;

ov = iw.*ih./ua;
ov(iw<=0 | ih<=0) = 0; %No overlap


