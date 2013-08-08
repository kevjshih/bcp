function [roc_sm roc] = test_part_detections_D(cls, D, part_scores, nms_th, ignore_dup)

% -------- Do Not Change ---------- 
areaThresh = 24*24; % this should stay fixed
ovThresh = 0.00000001; ;
% --------------------------------- %

if(~exist('ignore_dup', 'var'))
   ignore_dup = 0;
end

if(~exist('nms_th', 'var') || isempty(nms_th))
   nms_th = 0.5;
end

BDglobals;

for i = 1:length(D)
   [dk ids{i}] = fileparts(D(i).annotation.filename);
  
   scores = part_scores{i};
  
   if(size(scores,2)==9) % Repredicted box
      reg_inds = nms_center([scores(:, [5:8 end])]);
      reg = scores(reg_inds, [5:8]);
   else
      reg_inds = nms_center([scores(:, [1:4, end])]);
      reg = scores(reg_inds, [1:4]);
   end

   sc = scores(reg_inds, end);
   
   boxes_final{i} = reg;
   scores_final{i} = sc;
end


[roc_sm roc] = evaluate_center(D, boxes_final, scores_final, cls);
%res = evaluateDetections(D, im_dir, {cls}, {}, {}, ...
                             %boxes_final, scores_final, ovThresh, areaThresh, ignore_dup);

%roc_full = analyzeResultNew(res);

%roc_sub.ap = roc_full.ap;
%roc_sub.area = roc_full.area;
%roc_sub.area_2fp = roc_full.area_2fp;
%roc_sub.nfound = roc_full.nfound;
%roc_sub.nim = roc_full.nim;
%roc_sub.normAP = roc_full.normAP;
%roc_sub.npos = roc_full.npos;
%roc_sub.pct_found = roc_full.pct_found;

fprintf('%s: AP: %0.3f, NoDup AP: %0.3f\n', cls, roc_sm.ap, roc_sm.ap_nodup);


function [roc_sm roc] = evaluate_center(D, boxes, scores, cls)

Npos = 0;

for i = 1:length(D)
   gt = LMobjectboundingbox(D(i).annotation, cls);
   
   if(isempty(gt))
      tp = zeros(size(scores{i},1), 1); %any(ok(j, :));
      fp = ones(size(scores{i},1), 1); %any(ok(j, :));
      dup = zeros(size(scores{i},1), 1);
      ind = 1:length(dup);
   else
      [sc ind] = sort(scores{i}, 'descend');
      det = boxes{i}(ind, :);

      Npos = Npos + size(gt, 1);

      dup = zeros(length(sc), 1);
      tp = zeros(length(sc), 1);
      fp = zeros(length(sc), 1);
      found = zeros(1,size(gt,1));

      ok = contained_center(det, gt);
 
      for j = 1:length(sc)
         % TP if any detections are ok and not already found
         tp(j) = any(ok(j, :) & ~found); % & ~all(found(ok(j,:)));
         fp(j) = ~tp(j);
   
         if(fp(j))
            dup(j) = any(found(ok(j,:)));
         else
            found(ok(j,:)) = true;
         end
      end
   end


   % Now save results (Unsorting as necessary)
   all_dup{i}(ind) = dup;
   all_fp{i}(ind) = fp;
   all_tp{i}(ind) = tp;
end

tp = cat(2, all_tp{:})';
fp = cat(2, all_fp{:})';
dup = cat(2, all_dup{:})';
sc = cat(1, scores{:});

[sc_sort inds] = sort(sc, 'descend');

last = min(min([find(isinf(sc_sort)) inf]), max(find(tp(inds)==1)));

if(isempty(last))
   last = min(1e5, length(sc_sort));
end

last = min(1e5, last);

roc.conf = sc_sort(1:last);
roc.tp = cumsum(tp(inds(1:last)));
roc.fp = cumsum(fp(inds(1:last)));
roc.fp_nodup = cumsum(fp(inds(1:last)) & ~dup(inds(1:last)));

roc.p = roc.tp ./ (roc.tp + roc.fp);
roc.r = roc.tp / Npos;
roc.ap = VOCap(roc.r, roc.p);

roc.p_nodup = roc.tp ./ (roc.tp + roc.fp_nodup);
roc.ap_nodup = VOCap(roc.r, roc.p_nodup);
roc.Npos = Npos;


roc_sm.tp = sparse(tp(inds(1:last)));
%roc_sm.fp = sparse(fp(inds));
roc_sm.dup = sparse(dup(inds(1:last)));
roc_sm.conf = sc_sort(1:last);
roc_sm.ap = roc.ap;
roc_sm.Npos = Npos;
roc_sm.ap_nodup = roc.ap_nodup;
