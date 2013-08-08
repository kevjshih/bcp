function pick = nms(boxes, th)
% top = nms_fast(boxes, overlap)
% Non-maximum suppression. (FAST VERSION)
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.
% NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
% but an inner loop has been eliminated to significantly speed it
% up in the case of a large number of boxes
% Tomasz Maliseiwicz (tomasz@cmu.edu)

if isempty(boxes)
  top = [];
  return;
end

x1 = boxes(:,1);
y1 = boxes(:,2);
x2 = boxes(:,3);
y2 = boxes(:,4);
s = boxes(:,end);

[vals, I] = sort(s);

pick = s*0;
counter = 1;
while ~isempty(I)
  
  last = length(I);
  i = I(last);  
  pick(counter) = i;
  counter = counter + 1;
 
  center = 1/2*[boxes(i, [1 2]) + boxes(i, [3 4])]; 
 
%  xx1 = max(x1(i), x1(I(1:last-1)));
%  yy1 = max(y1(i), y1(I(1:last-1)));
%  xx2 = min(x2(i), x2(I(1:last-1)));
%  yy2 = min(y2(i), y2(I(1:last-1)));
   xin = (x1(I(1:last-1)) <= center(1)) & (x2(I(1:last-1)) >= center(1));
   yin = (y1(I(1:last-1)) <= center(2)) & (y2(I(1:last-1)) >= center(2));
 
   ok = yin & xin; 
  I([last; find(ok)]) = [];
end

pick = pick(1:(counter-1));
