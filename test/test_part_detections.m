function [recall, prec, ap] = test_part_detections(cls, ids, scores, regions, parts)

if(~exist('VOCopts', 'var'))
   VOCinit;
end

start = tic;
for i = 1:length(ids)
   if(toc(start)>5)
      fprintf('%d/%d\n', i, length(ids));
      start = tic;
   end
   rec=PASreadrecord(sprintf(VOCopts.annopath,ids{i}));
   clsinds=strmatch(cls,{rec.objects(:).class},'exact');
   diff=[rec.objects(clsinds).difficult];


   if(isempty(clsinds) && ~isempty(regions{i}))
      % Negative image: 2 types of NMS
      reg_inds = nms_v4([regions{i} scores{i}(:)], 0.5);

      reg = regions{i}(reg_inds,:);
      sc = scores{i}(reg_inds);

      if(isempty(parts))
         [dk pt_inds]  = unique(sc);
      else
         pts = parts{i}(reg_inds,:);

         pt_inds = nms_v4([pts sc(:)], 0.99); % suppress any identical parts
      end

      boxes_final{i} = reg(pt_inds,:);
      scores_final{i} = sc(pt_inds, :);
   else
      % Positive image, only worry about positive windows for now
      % Just find the highest scoring region that overlaps with GT
      todo = clsinds(~diff);

      bb = zeros(0, 4);
      sc = zeros(0, 1);

      if(~isempty(todo))
         overlaps = bbox_overlap(regions{i}, cat(1, rec.objects(todo).bbox));

         for gt = 1:length(todo)
            ok = find(overlaps(gt, :)>0.5);
            
            if(~isempty(ok) && any(~isinf(scores{i}(ok))))
               [dk best_ind] = max(scores{i}(ok));
               bb = [bb; regions{i}(ok(best_ind),:)];
               sc = [sc; scores{i}(ok(best_ind))];
            end
         end
      end
      boxes_final{i} = bb;
      scores_final{i} = sc;
   end
end

% create results file
fprintf('Writing results to %s\n', sprintf(VOCopts.detrespath,'comp3',cls));
fid=fopen(sprintf(VOCopts.detrespath,'comp3',cls),'w');

% Now write it
tic;
for i=1:length(ids)
    % display progress
    if toc>1
        fprintf('%s: test: %d/%d\n',cls,i,length(ids));
        drawnow;
        tic;
    end

    % compute confidence of positive classification and bounding boxes
   c = scores_final{i};
  BB = boxes_final{i}';
    % write to results file
    for j=1:length(c)
        fprintf(fid,'%s %f %f %f %f %f\n',ids{i},c(j),BB(:,j));
    end
end


[recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);
