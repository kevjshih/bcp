function [roc res] = test_given_cache(D, cached_scores1, cached_scores2, cls, show, do_nms)

% -------- Do Not Change ---------- %
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
th = 0.5;

[dk good] = LMquery(D, 'object.name', cls, 'exact');
% Select boxes with simple NMS
for i = 1:length(D)%randperm(length(D))
   if(~isempty(cached_scores1{i}.scores) && ismember(i, good))
         all_detections1 = [cached_scores1{i}.regions, cached_scores1{i}.scores];
         det_ind1 = nms_v4(all_detections1, th);
         detections1 = all_detections1(det_ind1, :);

         all_detections2 = [cached_scores2{i}.regions, cached_scores2{i}.scores];
         det_ind2 = nms_v4(all_detections2, th);
         detections2 = all_detections2(det_ind2, :);

         im = LMimread(D, i, im_dir);
         gtboxes = LMobjectboundingbox(D(i).annotation, cls);

         %for k = 1:1%min(5, numel(scores{i}))
         k = 1; % Top detection
         %%%%%%%%%%%%%% Scores 1
         figure(1);
            clf;
            imagesc(im);
            hold on;
            if(numel(gtboxes)>0 && any(bbox_overlap_mex(gtboxes, detections1(k,1:4))>0.5))
               obj_col = 'g';
            else
               obj_col = 'r';
            end
            draw_bbox(detections1(k, :), obj_col, 'linewidth', 3);

            for j = 1:size(cached_scores1{i}.part_boxes,2)/4
               draw_bbox(cached_scores1{i}.part_boxes(det_ind1(k), (j-1)*4 + [1:4]), ['--' col(mod(j, end)+1)]);
            end
         %%%%%%%%%%%%%% Scores 2 
         figure(2);
            clf;
            imagesc(im);
            hold on;
            if(numel(gtboxes)>0 && any(bbox_overlap_mex(gtboxes, detections2(k,1:4))>0.5))
               obj_col = 'g';
            else
               obj_col = 'r';
            end
            draw_bbox(detections2(k, :), obj_col, 'linewidth', 3);

            for j = 1:size(cached_scores2{i}.part_boxes,2)/4
               draw_bbox(cached_scores2{i}.part_boxes(det_ind2(k), (j-1)*4 + [1:4]), ['--' col(mod(j, end)+1)]);
            end
         % Done
            pause
      end

   [dk ids{i}] = fileparts(D(i).annotation.filename);
end

if(show==2)
show_top_detections(detection_list, D, cls);
end
if(1)
 res = evaluateDetections(D, im_dir, {cls}, {}, {}, ...
                             boxes, scores, th, areaThresh, ignore_dup);

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

figure(2)
[recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);


