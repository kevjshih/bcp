function [roc categories] = test_fgmr_precomputed(D, cls, nmsth, evaltype)

% -------- Do Not Change ---------- %
areaThresh = 24*24; % this should stay fixed
ovThresh = 0.50;
ovThreshPart = 0.25;
% --------------------------------- %

if(~exist('nmsth', 'var') || isempty(nmsth))
    nmsth = 0.5;
end

if(numel(nmsth)==2)
   ovThresh = nmsth(2);
   nmsth = nmsth(1);
end

if(~exist('evaltype', 'var') || isempty(evaltype))
   evaltype = 1;
end

ignoreDup = 0;

if(evaltype==1)
   usevoc = 0;
   dopoint = 0;
end

if(evaltype==2)
   usevoc = 1;
   dopoint = 0;
end

if(evaltype==3)
   usevoc = 0;
   dopoint = 1;
end

if(evaltype==4)
   usevoc = 0;
   dopoint = 0;
   ignoreDup = 1;
   ovThresh = 0.1;
end

if(dopoint==1)
   ovThresh = 0.00000001; % Requires that point be contained within box
   ignoreDup = 1;
end


for i = 1:length(D)
   D(i).annotation.folder = '';
end

BDglobals; % Defines im_dir

col = 'cmykrgb';


categories = {};

start = tic;
for i = 1:length(D)
   if(toc(start)>5)
      fprintf('%d/%d\n', i, length(D));
      start = tic;
   end
   [dk bn] = fileparts(D(i).annotation.filename);
   res = load(sprintf('data/detections_1000/%s_fgmr.mat', bn));

   if(isempty(categories))
      categories = res.detector_names;
        categories = strtok(categories, '_');
   end

   for j = find(ismember(categories, cls))%%1:length(categories)
       if(nmsth==0.5)
        boxes{i, j} = res.detections_nms{j}(:,1:4);
        scores{i,j} = res.detections_nms{j}(:,6);
       else
         inds = nms_v4(res.detections{j}(:, [1:4 6]), nmsth);
         boxes{i,j} = res.detections{j}(inds, 1:4);
         scores{i,j} = res.detections{j}(inds, 6);
       end

      if(dopoint)
         det = boxes{i,j};
         boxes{i,j} = repmat(1/2*(det(:,3:4) + det(:,1:2)), 1, 2) + repmat([0 0 1 1], size(det,1), 1);
      end
   end
   [dk ids{i}] = fileparts(D(i).annotation.filename);
end

categories = strtok(categories, '_');

if(~exist('cls', 'var'))
   todo = 1:length(categories)
else
   todo = find(ismember(categories, cls));
end

if(usevoc)
   VOCinit;
   VOCopts.testset = 'val';
end

for cind = todo
   cls = categories{cind};
   fprintf('Evaluating %s\n', cls);

   if(usevoc)
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
         c = scores{i, cind};
         BB = boxes{i, cind}';
         % write to results file
         for j=1:length(c)
            fprintf(fid,'%s %f %f %f %f %f\n',ids{i},c(j),BB(:,j));
         end
      end

      % close results file
      fclose(fid);

      figure(2)
      [recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);

      roc{cind}.r = recall;
      roc{cind}.p = prec;
      roc{cind}.ap = ap;
   else
      %try
         results{cind} = evaluateDetections(D, im_dir, {cls}, {}, {}, ...
                                   boxes(:,cind), scores(:,cind), ovThresh, areaThresh, ignoreDup);
      
         roc{cind} = analyzeResultNew(results{cind});
      %catch
      %   fprintf('error with %s\n', cls);
      %end
   end
end
