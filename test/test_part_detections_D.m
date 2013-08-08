function [recall, prec, ap, pos_prec roc] = test_part_detections_D(cls, D, cached_scores, part_scores, neg_weight)
%[recall, prec, ap, pos_prec, pos_prec] = test_part_detections_D(cls, D, cached_scores, part_ind)

if(~exist('neg_weight', 'var'))
   neg_weight = 1;
end

if(~exist('VOCopts', 'var'))
  VOCinit;
end

if(exist('set', 'var'))
  VOCopts.testset = set;
end

best_pos_score = cell(numel(D), 1);

neg_sc = [];
pos_sc = [];

for i = 1:length(D)
  
  [dk ids{i}] = fileparts(D(i).annotation.filename);
  
  if(isempty(cached_scores{i}.labels))
    continue;
  end
  
  if(~iscell(part_scores) || numel(part_scores)==1) % Just supplying an index into cached scores
    scores = cached_scores{i}.part_scores(:,part_scores);
  else
    scores = part_scores{i};
  end
  
  boxes = LMobjectboundingbox(D(i).annotation, cls);
  
  if(isempty(boxes))
    % Negative image: 2 types of NMS
    
    %      tic;reg_inds = nms_v4([cached_scores{i}.regions scores(:)], 0.5);toc;
    reg_inds = nms_v4([cached_scores{i}.regions scores(:)], 0.5);
    
    reg = cached_scores{i}.regions(reg_inds,:);
    sc = scores(reg_inds);
    
    [dk pt_inds]  = unique(sc, 'first'); % Suppress duplicate parts
    
    boxes_final{i} = reg(pt_inds,:);
    scores_final{i} = sc(pt_inds, :);
    
    neg_sc = [neg_sc; sc(pt_inds, :)];
  else
    % Positive image, only worry about positive windows for now
    % Just find the highest scoring region that overlaps with GT
    %      todo = clsinds;
    
    bb = zeros(0, 4);
    sc = zeros(0, 1);
    
    if(1)%~isempty(todo))
      overlaps = bbox_overlap(cached_scores{i}.regions, boxes);
      
      for gt = 1:size(boxes,1)
        ok = find(overlaps(gt, :)>0.5);
        
        if(~isempty(ok) && any(~isinf(scores(ok))))
          [dk best_ind] = max(scores(ok));
          bb = [bb; cached_scores{i}.regions(ok(best_ind),:)];
          sc = [sc; scores(ok(best_ind))];
          pos_sc = [pos_sc; scores(ok(best_ind))];
          best_pos_score{i}(gt) = scores(ok(best_ind));
        else
          pos_sc = [pos_sc; -inf];
          best_pos_score{i}(gt) = -inf; % Missed it!
        end
      end
    end
    boxes_final{i} = bb;
    scores_final{i} = sc;
  end
end

do_voc_ap = 0;

if(~do_voc_ap)
  roc = computeROC([neg_sc; pos_sc], [-ones(numel(neg_sc),1); ones(numel(pos_sc),1)], neg_weight);
  ap = VOCap(roc.r, roc.p);
  
  recall = roc.r;
  prec = roc.p;
else
  a = tic;
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
  
  fclose(fid);
  [recall,prec,ap]=VOCevaldet(VOCopts,'comp3',cls,true);
end

prec_smooth = cummax_mex(prec(end:-1:1));
prec_smooth = prec_smooth(end:-1:1);

if(nargout>=4) % Compute max precision for each example
  %hold on;
  confidences = sort(cat(1, scores_final{:}), 'descend');
  pos_prec = cell(length(D), 1);
  for i = 1:length(best_pos_score)
    if(isempty(best_pos_score{i}))
      continue;
    end
    
    
    for gt = 1:length(best_pos_score{i})
      if(isinf(best_pos_score{i}(gt)))
        pos_prec{i}(gt) = 0;
      else
        prec_ind = min(find(confidences<=best_pos_score{i}(gt)));
        pos_prec{i}(gt) = prec_smooth(prec_ind);
        
        %plot(recall(prec_ind), prec_smooth(prec_ind), 'xr');
      end
    end
  end
  
  %hold off;
end
