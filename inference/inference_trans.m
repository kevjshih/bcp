function [hyp feat_data] = inference(input, model, regions, cached_scores)
% Search over L/R flips, subcells, etc.
% Input
%   a) YxXx3 image
%   b) input.feat
%      input.scales
% model ...
% regions ...
% Cached Data (for each region)
%  cached_scores
Nreg = size(regions,1);

sbin = model.sbin;
interval = model.interval;

padx = ceil(model.maxsize(2)/2+1);
pady = ceil(model.maxsize(1)/2+1);

if(~isfield(model, 'rotation'))
   model.rotation = 0;
end

if(~isstruct(input)) % it's an image!
   [feat, scales, imsizes tmp] = IEfeatpyramid_trans(input, sbin, interval, model.shift, model.rotation); % model.rot); % Currently not handling rotations (need to figure out how to represent rotated boxes)

   for s = 1:numel(feat) % pad it (will be cached this way)
      feat{s} = padarray(feat{s}, [pady padx 0], 0);
   end

   if(nargout>=2) % return precomputed features
      [feat_data.feat feat_data.scales feat_data.imsizes] = deal(feat, scales, imsizes);
   end
else
   [feat, scales, imsizes] = deal(input.feat, input.scales, input.imsizes);
   if(nargout>=2)
      feat_data = input;
   end
end

[xs0 ys0] = meshgrid(1:size(feat{1},2), 1:size(feat{1},1));

if(~exist('cached_scores', 'var'))
   cached_scores = zeros(numel(regions), numel(model.cached_weight));
end

iscached = [model.part.computed];

hyp = init_hyp(cached_scores, Nreg, model);

part_inds = find(~iscached);

for pi = 1:model.num_parts
   if(iscached(pi))
      continue;
   end
   i = pi;%part_inds(pi);
   pm = model.part(pi);

   best_scores = -inf(size(regions,1), 1);
   if(isfield(pm, 'spat_w') && ~isempty(pm.spat_w))
      best_loc = ones(size(regions,1), 7 + 3); % [x y s flip xs yx rot (xb yb sb)];
      % Smooth spatial model
      % f = [0.5 1 0.5]'*[0.5 1 0.5];
      % pm.spat_w = filter2(f, pm.spat_w);
   else
      %best_loc = ones(size(regions,1), 3); % [x y s];
      best_loc = ones(size(regions,1), 7); % [x y s flip xs yx rot];
   end
   best_box = ones(size(regions,1), 4); % [x y s flip xs yx rot];

   if(~isfield(pm, 'spat_const') || isempty(pm.spat_const));
      const = [0 1 0.8 1 0 1];
      %const = [0 1 0 1 0.8 1]; % For now, require high overlap with region
   else
      const = pm.spat_const;
   end

   if(isfield(pm, 'whiten'))
       filter0 = pm.filter;
       pm.filter = whiten_filter(filter0, pm.whiten.W);
       pm.bias = pm.bias - whiten_bias(filter0, pm.whiten.W, pm.whiten.mu);
   end
       
       
   if(isfield(pm, 'reference_box') && ~isempty(pm.reference_box))
      regions0 = regions;
      % Transform regions;
      sc_all = sqrt(sum((regions(:, [3 4]) - regions(:, [1 2])).^2, 2));
      cent_all = 1/2*(regions(:, [3 4]) + regions(:, [1 2]));

      regions = [cent_all cent_all] + bsxfun(@times, pm.reference_box, sc_all);
      regions_flipped = [cent_all cent_all] + bsxfun(@times, [-1 1 -1 1], bsxfun(@times, pm.reference_box([3 2 1 4]), sc_all));
   else
      regions_flipped = regions;
   end


   for level = 1:length(scales)
      for i_x_sh = 1:length(model.shift)
         x_shift = model.shift(i_x_sh);
         for i_y_sh = 1:length(model.shift)
            y_shift = model.shift(i_y_sh);
            for i_rot = 1:length(model.rotation)
               featr = feat{level, i_x_sh, i_y_sh, i_rot};
               scale = model.sbin/scales(level);

               for trans_LR = 1:2
                  if(trans_LR==1) % Standard
                     filter = {pm.filter};
                     cur_reg = regions;
                  else
                     filter = {flipfeat(pm.filter)};
                     cur_reg = regions_flipped;
                  end

                  % Compute scores
                  rootmatch_cell = fconv(double(featr(:,:,1:size(filter{1},3))), filter, 1, 1);
                  rootmatch = rootmatch_cell{1} + pm.bias;
                  Sx = size(rootmatch,2);
                  Sy = size(rootmatch,1);
                  xs = xs0(1:Sy, 1:Sx);
                  ys = ys0(1:Sy, 1:Sx);
    
                  % Prune candidates: 
                  min_score = min(best_scores);
                  ok_score = rootmatch>min_score; % No need to consider these...
                  xsub = xs(ok_score);
                  ysub = ys(ok_score);
                  boxes = rootbox_trans(xsub, ysub, scale, [x_shift, y_shift]/model.sbin, model.rotation(i_rot), padx, pady, pm.size(1:2), size(input));
                  scores = rootmatch(ok_score);
      
                  if(isfield(pm, 'spat_w') && ~isempty(pm.spat_w)) % Search over spatial locations
                     if(trans_LR==1)
                        spat_w = pm.spat_w;
                      else % Flip it
                        spat_w = pm.spat_w(:, end:-1:1);
                      end
                  %[best_score pos] = get_best_part_spat_new(regions, boxes, scores, pm.spat_w, pm.scal_w, const(1), const(2), const(3), const(4), const(5), const(6));
                     [best_score pos] = get_best_part_spat_new(cur_reg, boxes, scores, model.spat_weight*spat_w, model.spat_weight*pm.scal_w, const(1), const(2), const(3), const(4), const(5), const(6));
                     [best_scores ind] = max([best_scores, best_score], [], 2);
                     updated = ind==2;
                     best_loc(updated, :) = [xsub(pos(updated, 1)), ysub(pos(updated, 1)), repmat([level trans_LR i_x_sh i_y_sh i_rot], sum(updated), 1), pos(updated, 2:4)];
                     %                           1                   2                                3      4       5      6    7                          8 9 10
                     best_box(updated, :) = boxes(pos(updated,1), :);
                  else
                     [best_score pos] = get_best_part(cur_reg, boxes, scores, const(1), const(2), const(3), const(4), const(5), const(6));
                     [best_scores ind] = max([best_scores, best_score], [], 2);
                     updated = ind==2;
                     best_loc(updated, :) = [xsub(pos(updated)), ysub(pos(updated)), repmat([level trans_LR i_x_sh i_y_sh i_rot], sum(updated), 1)];
                     %                              1                   2                      3      4       5      6    7
                     best_box(updated, :) = boxes(pos(updated,1), :);
                  end
               end % trans_LR
            end % i_rot
         end % i_y_sh
      end % i_x_sh
   end % level

   % Not too lazy anymore
   % Recompute box locations, because I was too lazy to cache them before ...
   % best_box = rootbox_trans(best_loc(:, 1), best_loc(:,2), model.sbin./scales(best_loc(:,3)), model.shift([best_loc(:,5) best_loc(:,6)])/model.sbin, model.rotation(best_loc(:,7)), padx, pady, pm.size(1:2), size(input));
   hyp = update_hyp(hyp, best_scores, best_loc, best_box, i);    
end

hyp = prune_hyp(hyp, model);


function hyp = prune_hyp(hyp, model)
   if(model.thresh>-inf)
      final_scores = [hyp.final_score];
      remove = final_scores<model.thresh | isinf(final_scores);

      hyp(remove) = [];
   end

function hyp = init_hyp(init_score, Nreg, model)

Npart = model.num_parts;
hyp = repmat(struct('computed', [], 'score', [], 'loc', [], 'bbox', []), Nreg,1);

toadd = Npart;% - sum([model.part.computed]);

for r = 1:length(hyp)
   hyp(r).region = r;
   hyp(r).computed = zeros(toadd, 1);
   hyp(r).score = zeros(toadd, 1);
   hyp(r).loc = zeros(toadd, 3);
   hyp(r).bbox = zeros(toadd,4);
   hyp(r).cached_score = init_score(r,:);
   hyp(r).final_score = model.cached_weight*init_score(r,:)';
end


function hyp = update_hyp(hyp, scores, loc, bbox, pind, bins)
% Probably not the most efficient...
if(nargin==2) % Just updating the bias
   for r = 1:length(hyp)
      hyp(r).final_score = hyp(r).final_score + scores;
   end
else
   for r = 1:length(scores)
      hyp(r).region = r;
      hyp(r).computed(pind) = 1;
      hyp(r).score(pind) = scores(r);
      hyp(r).loc(pind,1:size(loc,2)) = loc(r,:);
      hyp(r).bbox(pind,:) = bbox(r,:);
      hyp(r).final_score = hyp(r).final_score + scores(r);
   end
end

function boxes = rootbox(x, y, scale, padx, pady, rsize)
x1 = (x(:)-padx).*scale(:)+1;
y1 = (y(:)-pady).*scale(:)+1;
x2 = x1 + rsize(2).*scale(:) - 1;
y2 = y1 + rsize(1).*scale(:) - 1;

boxes = [x1 y1 x2 y2];

