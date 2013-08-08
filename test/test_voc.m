function [roc res] = test_given_cache(D, cached_scores, cls, th, ignore_dup, show, do_nms)
%[roc res] = test_given_cache(D, cached_scores, cls, th, ignore_dup, show, do_nms)
% th: [eval nms]
% -------- Do Not Change ---------- 
areaThresh = 24*24; % this should stay fixed
ovThresh = 0.50;
ovThreshPart = 0.25;
% --------------------------------- %

if(~exist('ignore_dup','var'))
   ignore_dup = 0;
end

if(~exist('show','var'))
   show = 0;
end

if(~exist('do_nms', 'var'))
   do_nms = 1;
end

for i = 1:length(D)
   D(i).annotation.folder = '';
end

BDglobals; % Defines im_dir

col = 'cmykrgb';

fprintf('Using %f overlap for NMS\n', th(end));
[dk good] = LMquery(D, 'object.name', cls, 'exact');
% Select boxes with simple NMS
for i = 1:length(D)%randperm(length(D))
   if(~isempty(cached_scores{i}.scores))
      if(do_nms)
         all_detections = [cached_scores{i}.regions, cached_scores{i}.scores];
         det_ind = nms_v4(all_detections, th(end));
      else
         all_detections = [cached_scores{i}.regions, cached_scores{i}.nms_scores];
         det_ind = 1:size(all_detections,1);
      end

      detections = all_detections(det_ind, :);
      boxes{i} = detections(:,1:4);
      scores{i} = detections(:, end);
      detection_list{i} = detections;
      if(show==1&&ismember(i, good))
         im = LMimread(D, i, im_dir);

         for k = 1:1%min(5, numel(scores{i}))
            clf;
            imagesc(im);
            hold on;
            draw_bbox(detections(k, :), 'linewidth', 3);

            for j = 1:size(cached_scores{i}.part_boxes,2)/4
               draw_bbox(cached_scores{i}.part_boxes(det_ind(k), (j-1)*4 + [1:4]), ['--' col(mod(j, end)+1)]);
            end
            pause
         end
      end

      if(show==2&&ismember(i, good))
         im = LMimread(D, i, im_dir);
         gtboxes = LMobjectboundingbox(D(i).annotation, cls);

         for k = 1:min(5, numel(scores{i}))
            clf;
            imagesc(im);
            hold on;
            if(numel(gtboxes)>0 && any(bbox_overlap_mex(gtboxes, detections(k,1:4))>0.5))
               obj_col = 'g';
            else
               obj_col = 'r';
            end
            draw_bbox(detections(k, :), obj_col, 'linewidth', 3);

            for j = 1:size(cached_scores{i}.part_boxes,2)/4
               draw_bbox(cached_scores{i}.part_boxes(det_ind(k), (j-1)*4 + [1:4]), ['--' col(mod(j, end)+1)]);
            end
            pause
         end
      end
   else
      boxes{i} = zeros(0, 4);
      scores{i} = zeros(0,1);
   end

   [dk ids{i}] = fileparts(D(i).annotation.filename);
end

if(show==3)
show_top_detections(detection_list, D);
end
if(0)
 fprintf('Using %f overlap for evaluation\n', th(1));
 res = evaluateDetections(D, im_dir, {cls}, {}, {}, ...
                             boxes, scores, th(1), areaThresh, ignore_dup);

roc = analyzeResultNew(res);

return
end

addpath(genpath(PASCALDIR));
% Do pascal evaluation
VOCinit;
VOCopts.testset = 'val';

% create results file
fprintf('Writing results to %s\n', sprintf(VOCopts.detrespath,'comp3',cls));
fid=fopen(sprintf(VOCopts.detrespath,'comp3',cls),'w');

% apply detector to each image
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end

    % compute confidence of positive classification and bounding boxes
   c = scores{i};
  BB = boxes{i}';
    % write to results file
    for j=1:length(c)
        fprintf(fid,'%s %f %f %f %f %f\n',ids{i},c(j),BB(:,j));
    end
end

% close results file
fclose(fid);

VOCopts.minoverlap = th(1);
figure(2)
[roc.recall,roc.prec,roc.ap]=VOCevaldet_mod(VOCopts,'comp3',cls,true, 1-ignore_dup);

fprintf('%s: %f AP at %f overlap\n', cls, roc.ap, th(1));

drawnow;
