function [hyp feat_data] = inference(input, model, regions, cached_scores)
% Input
%   a) YxXx3 image
%   b) input.feat
%      input.scales
% model ...
% regions ...
% Cached Data (for each region)
%  cached_scores
if(isfield(model, 'do_transform') && model.do_transform==1)
   if(isfield(model, 'shift') && ~isempty(model.shift))
      [hyp feat_data] = inference_trans(input, model, regions, cached_scores);
   else
      [hyp feat_data] = inference_transf(input, model, regions, cached_scores);
   end
   return;
end

Nreg = size(regions,1);

sbin = model.sbin;
interval = model.interval;

padx = ceil(model.maxsize(2)/2+1);
pady = ceil(model.maxsize(1)/2+1);

if(~isstruct(input)) % it's an image!
   [feat, scales] = IEfeatpyramid(input, sbin, interval);

   if(nargout>=2) % return precomputed features
      [feat_data.feat feat_data.scales] = deal(feat, scales);
   end
else
   [feat, scales] = deal(input.feat, input.scales);
   if(nargout>=2)
      feat_data = input;
   end
end

for s = 1:length(feat) % pad it
   feat{s} = padarray(feat{s}, [pady padx 0], 0);
end

[xs0 ys0] = meshgrid(1:size(feat{1},2), 1:size(feat{1},1));

if(~exist('cached_scores', 'var'))
   cached_scores = zeros(numel(regions), 1);
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
      best_loc = ones(size(regions,1), 6); % [x y s xb yb sb];
      % Smooth spatial model
%      f = [0.5 1 0.5]'*[0.5 1 0.5];
%      pm.spat_w = filter2(f, pm.spat_w);
   else
      best_loc = ones(size(regions,1), 3); % [x y s];
   end

   if(~isfield(pm, 'spat_const') || isempty(pm.spat_const));
      const = [0 1 0.8 1 0 1];
      %const = [0 1 0 1 0.8 1]; % For now, require high overlap with region
   else
      const = pm.spat_const;
   end

   for level = 1:length(feat)
      featr = feat{level};
      scale = model.sbin/scales(level);

      % Compute scores
      rootmatch_cell = fconv(featr(:,:,1:31), {pm.filter}, 1, 1);
      rootmatch = rootmatch_cell{1};
      Sx = size(rootmatch,2);
      Sy = size(rootmatch,1);
      xs = xs0(1:Sy, 1:Sx);
      ys = ys0(1:Sy, 1:Sx);
    
      if(1) % Fast version (maybe not that helpful...)
         % Prune candidates: 
         min_score = min(best_scores);
         ok_score = rootmatch>min_score; % No need to consider these...
         xsub = xs(ok_score);
         ysub = ys(ok_score);
         boxes = rootbox(xsub, ysub, scale, padx, pady, pm.size(1:2));
         scores = rootmatch(ok_score);
      
         if(isfield(pm, 'spat_w') && ~isempty(pm.spat_w)) % Search over spatial locations
            %[best_score pos] = get_best_part_spat_new(regions, boxes, scores, pm.spat_w, pm.scal_w, const(1), const(2), const(3), const(4), const(5), const(6));
            [best_score pos] = get_best_part_spat_new(regions, boxes, scores, model.spat_weight*pm.spat_w, model.spat_weight*pm.scal_w, const(1), const(2), const(3), const(4), const(5), const(6));
            [best_scores ind] = max([best_scores, best_score], [], 2);
            updated = ind==2;
            best_loc(updated, :) = [xsub(pos(updated, 1)), ysub(pos(updated, 1)), repmat(level, sum(updated), 1) pos(updated, 2:4)];
         else
            [best_score pos] = get_best_part(regions, boxes, scores, const(1), const(2), const(3), const(4), const(5), const(6));
            [best_scores ind] = max([best_scores, best_score], [], 2);
            updated = ind==2;
            best_loc(updated, :) = [xsub(pos(updated)), ysub(pos(updated)), repmat(level, sum(updated), 1)];
         end
      else % Slow readable version
         % Find candidates for each region 
         boxes = rootbox(xs, ys, scale, padx, pady, pm.size(1:2));
      
         ok = double(check_overlap(boxes, regions));
         ok(ok==0) = -inf;
         ok(ok==1) = 0;
         [best_score, pos] = max(bsxfun(@plus, ok, rootmatch(:))', [], 2);
       
         %[a b] = get_best_part(regions, boxes, rootmatch(:), [], [], [], [], 0.25);

         % Update scores
         [best_scores ind] = max([best_scores, best_score], [], 2);
         updated = ind==2;
         best_loc(updated, :) = [xs(pos(updated)), ys(pos(updated)), repmat(level, sum(updated), 1)];
      end
   end

   best_box = rootbox(best_loc(:, 1), best_loc(:,2), model.sbin./scales(best_loc(:,3)), padx, pady, pm.size(1:2));

   hyp = update_hyp(hyp, best_scores + model.part(i).bias, best_loc, best_box, i);    
end

%hyp = update_hyp(hyp, model.bias);
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
   hyp(r).cached_score = init_score(r);
   hyp(r).final_score = model.cached_weight*init_score(r);
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

function ok = check_overlap(boxes, regions)

%ok = bbox_overlap(regions, boxes,0.25)>0.25;

ok = bbox_overlap_mex(boxes, regions, 0.25)>0.25;
% = bbox_overlap_mex(boxes, regions)>0.25;

