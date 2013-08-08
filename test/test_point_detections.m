function [roc res] = test_point_detections(D, cached_scores, cls, th, column)
%function [roc res] = test_point_detections(D, cached_scores, cls, th, column)

% -------- Do Not Change ---------- %
areaThresh = 24*24; % this should stay fixed
ovThresh = 0.50;
ovThreshPart = 0.25;
% --------------------------------- %

if(~exist('column', 'var'))
   column = 0;
end


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
         if(column==0) % Use overall score
            all_detections = [cached_scores{i}.regions, cached_scores{i}.scores];
            det_ind = nms_v4(all_detections, th(end));
         else
            all_detections = [double(cached_scores{i}.part_boxes(:, 4*(column-1) + [1:4])), cached_scores{i}.part_scores(:, column)];
            det_ind = nms_v4(all_detections, th(end));
         end
      else
         all_detections = [cached_scores{i}.regions, cached_scores{i}.nms_scores];
         det_ind = 1:size(all_detections,1);
      end

      detections = all_detections(det_ind, :);
      boxes{i} = repmat(1/2*(detections(:,3:4) + detections(:,1:2)), 1, 2) + repmat([0 0 1 1], size(detections,1), 1);
      scores{i} = detections(:, end);
      detection_list{i} = detections;
   else
      boxes{i} = zeros(0, 4);
      scores{i} = zeros(0,1);
   end

   [dk ids{i}] = fileparts(D(i).annotation.filename);
end

if(show==3)
show_top_detections(detection_list, D);
end

th(1) = 0.00000001; % Requires that point be contained within box
ignore_dup = 1;

if(1)
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

figure(2)
[recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);


