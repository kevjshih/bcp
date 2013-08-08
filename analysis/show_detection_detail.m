function show_detection_detail(D, cached_scores, model, th)

addpath('~/prog/voc-release4');

BDglobals;
VOCinit;
classes = VOCopts.classes;

fgmr = load('~/prog/voc-release4/VOC2009/horse_final.mat');

[D inds] = LMquery(D, 'object.name', model.cls);

cached_scores = cached_scores(inds);

perm = randperm(length(D));

if(~exist('th','var'))
   th = -inf;
end

columns = model.learner.model.getColumnsUsed+1;
%best_columns = columns(1:5);
%best_columns = [17 29 27 39 11 2];
%threshes = [-0.8725688 -0.66701907 -0.7019135 -0.6128856 -1.0331464  -0.7961178];
best_columns = [20 37 32 24 20 10];
threshes = -ones(size(best_columns));



colors = 'bgrcmyk';

for i = perm
   % To Display: 
   % 1) Positive parts
   % 2) Top Detections
   inds = find(cached_scores{i}.scores>th);
   top_det = inds(nms_v4([cached_scores{i}.regions(inds,:) cached_scores{i}.scores(inds)], 0.5));

   im = imread(fullfile(im_dir, D(i).annotation.filename));
   clf;
   subplot(2,2,1)
   imagesc(im);
   axis image; axis off;
   hold on;
   
   Nfeat = size(cached_scores{i}.part_scores,2) + size(cached_scores{i}.region_score,2) +  1;
   contribution = get_score_contribution(cached_scores{i}.part_scores(top_det,:), model.learner, Nfeat);   
   part_scores = cached_scores{i}.part_scores(top_det,:);
%   [det_ind  part_ind] = find(contribution>0);
   for dind = 1:size(contribution,1)
   for pind0 = 1:length(best_columns)%length(part_ind)
      pind = best_columns(pind0);
      if(part_scores(dind,pind)>threshes(pind0))
         part_box = double(cached_scores{i}.part_boxes(top_det(dind), [1:4]+((pind)-1)*4));
         draw_bbox(part_box, [colors(mod(dind-1,7)+1) '-'], 'linewidth', (10-min(dind,9))/3);
         text(part_box(1), part_box(2), sprintf('%d:%.2f',(pind), contribution(dind, (pind))), 'backgroundcolor', 'g');
      end
   end
   end

   % Show top detections
   subplot(2,2,2)
   imagesc(im);
   axis image; axis off;
   hold on;

   best_det = double(cached_scores{i}.regions(top_det,:));
   draw_bbox(best_det, 'linewidth', 3);
   for j = length(top_det):-1:1
      text(best_det(j, 1), best_det(j, 2), sprintf('%d:%f', j, cached_scores{i}.scores(top_det(j))), 'backgroundcolor','g');
   end

   subplot(2,2,4)
   
   [dets, boxes] = imgdetect(im, fgmr.model, -1);
   top = nms(dets, 0.5);
   imagesc(im);
   axis image
   axis off
   hold on;
   if(~isempty(top))
      boxes = reduceboxes(fgmr.model, boxes(top,:));
      best_det = boxes(:, [1:4 end]);
      draw_bbox(best_det(:, 1:4), 'linewidth', 3);
      for j = size(best_det,1):-1:1
         text(best_det(j, 1), best_det(j, 2), sprintf('%d:%f', j, best_det(j, end)), 'backgroundcolor','g');
      end
   end

   pause 
end


function contribution = get_score_contribution(part_scores, boost_learner, Nfeat)

%feats = [cached_scores.part_scores cached_scores.region_score (1:Nreg)'];

% Find the contribution of each part:
for i = 1:size(part_scores,2)
   featT = -inf*ones(size(part_scores,1)+1, Nfeat);
   featT(:, i) = [-inf; part_scores(:, i)];

   contributionT = boost_classify(featT, boost_learner);
   contribution(:, i) = reshape(contributionT(2:end), [], 1) - contributionT(1);
end


