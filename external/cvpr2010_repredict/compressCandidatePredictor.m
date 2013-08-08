function pred = compressCandidatePredictor(pred, clusterov)

% merge significantly overlapping boxes
for k = 1:numel(pred)
  w = 100; h = 100; cx = 50; cy = 50; % fake bbox
  dx = pred(k).offset(:, 1); dy = pred(k).offset(:, 2);
  dsx = pred(k).offset(:, 3);  dsy = pred(k).offset(:, 4);
  sx = dsx*w;  sy = dsy*h;
  x = cx + dx*w; y = cy + dy*h;
  bbox = [x-sx/2 y-sy/2 x+sx/2 y+sy/2];  
  
  [keep, group] = bboxNonMaxSuppression(bbox, pred(k).w, clusterov);
  
  nkeep = sum(keep);
  pred(k).offset2 = zeros(nkeep, 4);
  pred(k).w2 = zeros(nkeep, 1);
  pred(k).n2 = nkeep;
  k_ind = find(keep);
  for k2 = 1:nkeep
      gind = group==k2;
      pred(k).offset2(k2, :) = pred(k).offset(k_ind(k2), :);
      pred(k).w2(k2) = sum(pred(k).w(gind));
  end  
 disp(num2str([pred(k).n pred(k).n2]))
 
 figure(1), hold off, plot(bbox(keep, [1 1 3 3 1])', bbox(keep, [2 4 4 2 2])')
end 