function [cached_scores counts roc rej_thresh acc_thresh fp masks] = prune_cascade_set(D, cached_scores, ...
                                                    cls)

    %roc = test_given_cache(D, cached_scores, cls, [0.1, 0.5], 1);
roc = test_point_detections(D, cached_scores, cls, [0.5]);
% Early rejection threshold:
min_recall = 0.99*roc.pct_found;   %ceil(0.99*(length(all_missed)-sum(all_missed))); % Get almost everything that can be found
rej_ind = min(find(roc.r>=min_recall));
rej_thresh = roc.conf(rej_ind);
fp = roc.fp(rej_ind);
if(isempty(rej_ind))
   rej_thresh = -inf;
end

min_prec = 0.995; %
acc_ind = max(find(roc.p>=min_prec));
acc_thresh = roc.conf(acc_ind);

if(isempty(acc_ind))
   acc_thresh = inf;
end

% Now prune out the examples!
accepted = 0;
rejected = 0;

BDglobals;

for i = 1:length(cached_scores)
   % 
   if(isempty(cached_scores{i}.labels))
      continue;
   end
   boxes = LMobjectboundingbox(D(i).annotation, cls);
   ok_ind = ones(numel(cached_scores{i}.labels), 1);

   if(~isempty(boxes)) % Some object found
      overlaps = bbox_overlap(cached_scores{i}.regions, boxes);
      
      for o = 1:size(boxes,1)
         ok = overlaps(o, :)>=0.5;

         if(max(cached_scores{i}.scores(ok))>=acc_thresh)
            accepted = accepted + 1;
            ok_ind(overlaps(o,:)>=0.4) = 0; % Remove anything that
                                            % has relatively high
                                            % overlap with 

         end
      end
   end
   acc = (ok_ind == 0);
   rej = cached_scores{i}.scores<=rej_thresh;
   masks(i).acc = acc;
   masks(i).rej = rej;
   %im = imread(fullfile(im_dir, D(i).annotation.filename));
   %clf
   %imagesc(im);
   %hold on;
   %draw_bbox(cached_scores{i}.regions(~rej & ok_ind, :), 'g');
   %draw_bbox(cached_scores{i}.regions(rej, :), 'r');
   %draw_bbox(cached_scores{i}.regions(~ok_ind, :), 'b');

   rejected = rejected + sum(rej);
   ok_ind(rej) = 0;
   cached_scores{i} = prune_cached_scores(cached_scores{i}, ok_ind==1);

   %pause 
end

fprintf('Accepted %d and rejected %d\n', accepted, rejected);
counts = [accepted rejected];
