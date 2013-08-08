function [boxes_out flipped_out locs_out] = inference(input, model, do_nms_iou)
% Search over L/R flips, subcells, etc.
% Input
%   a) YxXx3 image
%   b) input.feat
%      input.scales
% model ...
% regions ...
% Cached Data (for each region)
%  cached_scores


if(~exist('do_nms_iou', 'var'))
    do_nms_iou = 0;
end

sbin = model.sbin;
interval = model.interval;

padx = ceil(model.maxsize(2)/2+1);
pady = ceil(model.maxsize(1)/2+1);

if(~isfield(model, 'rotation'))
   model.rotation = 0;
end

if(~isfield(model, 'shift'))
    model.shift = 0;
end

if(~isstruct(input)) % it's an image!
   [feat, scales, imsizes tmp] = IEfeatpyramid_trans(input, sbin, interval, model.shift, model.rotation); % model.rot); % Currently not handling rotations (need to figure out how to represent rotated boxes)

   if(nargout>=2) % return precomputed features
      [feat_data.feat feat_data.scales feat_data.imsizes] = deal(feat, scales, imsizes);
   end
else
   [feat, scales, imsizes] = deal(input.feat, input.scales, input.imsizes);
   if(nargout>=2)
      feat_data = input;
   end
end

for s = 1:numel(feat) % pad it
   feat{s} = padarray(feat{s}, [pady padx 0], 0);
end

[xs0 ys0] = meshgrid(1:size(feat{1},2), 1:size(feat{1},1));


iscached = [model.part.computed];
parts_todo = find(~iscached);



for pi = 1:model.num_parts
   if(iscached(pi))
      continue;
   end
   i = pi;
   pm = model.part(pi);
   all_boxes = {};
   all_scores = {};
   all_locs = {};

   filters = {pm.filter, flipfeat(pm.filter)};

   for level = 1:length(scales)
      score_map = [];
      score_ind = [];
      scale = model.sbin/scales(level);

         i_x_sh = 1; i_y_sh = 1; i_rot = 1;
         featr = feat{level, i_x_sh, i_y_sh, i_rot};

         rootmatch_cell = fconv(featr(:,:,1:31), filters, 1, 2); % numbers indicate start and finish index of filters
         %rootmatch = rootmatch_cell{1};

      for trans_LR = 1:2
         if(trans_LR==1) % Standard
            filter = {pm.filter};
         else
            filter = {flipfeat(pm.filter)};
         end

         rootmatch = rootmatch_cell{trans_LR};

         % Compute scores
         %i_x_sh = 1; i_y_sh = 1; i_rot = 1;
         %featr = feat{level, i_x_sh, i_y_sh, i_rot};

         %rootmatch_cell = fconv(featr(:,:,1:31), filter, 1, 1);
         %rootmatch = rootmatch_cell{1};


         if(isempty(score_map)) % This is the first and current best
            Sx = size(rootmatch,2);
            Sy = size(rootmatch,1);
            xs = xs0(1:Sy, 1:Sx);
            ys = ys0(1:Sy, 1:Sx);

            score_ind = ones(size(xs,1), size(xs,2));
            score_map = rootmatch;

         else
            [best_score best_ind] = max(cat(3, score_map, rootmatch), [], 3);

            score_map = best_score;
            score_ind(best_ind==2) = trans_LR;
         end
      end % trans_LR
      
      all_boxes{level} = rootbox_trans(xs(:), ys(:), scale, [0 0], 0, padx, pady, pm.size(1:2), size(input));
      all_scores{level} = score_map(:);
      all_locs{level} = [xs(:), ys(:), repmat(level, numel(xs), 1), score_ind(:), repmat([1 1 1], numel(xs), 1)] ;
   end % level

   all_boxes = cat(1, all_boxes{:});
   all_scores = cat(1, all_scores{:});   
   all_locs = cat(1, all_locs{:});
   % Now do NMS
   if(do_nms_iou)
    inds = nms_cent_mex([all_boxes, all_scores]);
   else
    inds = nms_v4_fast([all_boxes, all_scores], 0.5);
   end
       
   boxes_out{pi} = [all_boxes(inds, :) all_scores(inds, :) + pm.bias];
   flipped_out{pi} = all_locs(inds, 4)-1; % Indicates if it was flipped
   locs_out{pi} = all_locs(inds, :); % Indicates if it was flipped
end
