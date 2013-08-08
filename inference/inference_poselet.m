function [hyp feat_data] = inference_poselet(input, model, regions, cached_scores)

feat_data = [];
% Since they're already using global variables, I'll do it too
global g_last_sz g_hog_blocks;

g_last_sz = [0 0 0];


Nreg = size(regions,1);

sbin = model.sbin;
interval = model.interval;

padx = ceil(model.maxsize(2)/2+1);
pady = ceil(model.maxsize(1)/2+1);

if(~isfield(model, 'rotation'))
   model.rotation = 0;
end

fprintf('Computing features\n');
phog=image2phog(input);
fprintf('Computing flipped features\n');
phog_flip =image2phog(input(:, end:-1:1, :)); % Lazy way
fprintf('Done\n');

if(~exist('cached_scores', 'var'))
   cached_scores = zeros(numel(regions), numel(model.cached_weight));
end

iscached = [model.part.computed];

hyp = init_hyp(cached_scores, Nreg, model);

part_inds = find(~iscached);

% Refactored code, setup all temporary data
for pi = 1:model.num_parts
   if(iscached(pi))
      continue;
   end
   i = pi;%part_inds(pi);
   pm = model.part(pi);

   best_scores{i} = -inf(size(regions,1), 1);
   best_loc{i} = ones(size(regions,1), 7); % [x y s flip xs yx rot];
   best_box{i} = ones(size(regions,1), 4); % [x y s flip xs yx rot];

   if(~isfield(pm, 'spat_const') || isempty(pm.spat_const));
      const_all{i} = [0 1 0.8 1 0 1]; % For now, require high overlap with region
   else
      const_all{i} = pm.spat_const;
   end
end


for level = 1:length(phog.hog)    
   for trans_LR = 1:2
      g_hog_blocks = [];
      num_blocks = 0; % Recompute
      sz = [0 0 0];
      
      for pi = 1:model.num_parts
         if(iscached(pi))
            continue;
         end

         const = const_all{pi};

         if(trans_LR==1) % Standard
            %filter = {pm.filter};
            curphog = phog;
         else
            %filter = {flipfeat(pm.filter)};
            curphog = phog_flip;
         end

         % Compute scores
         [boxes scores g_hog_blocks num_blocks sz] = detect_poselets_sc(curphog.hog{level}, model.part(pi), g_hog_blocks, num_blocks, sz); % This isn't the most efficient way, but feature computation is already slooow
         boxes = boxes/curphog.img_scale;
   
         if(trans_LR==2)
            boxes = flip_box(boxes, size(input));
         end
      
         scores = double(scores);
      
         [best_score pos] = get_best_part(regions, boxes, scores, const(1), const(2), const(3), const(4), const(5), const(6));
         [best_scores{pi} ind] = max([best_scores{pi}, best_score], [], 2);
         updated = ind==2;
         best_loc{pi}(updated, :) = [repmat([0 0 level trans_LR 0 0 0], sum(updated), 1)];
         %                               1    3       4     5 6 7
         best_box{pi}(updated, :) = boxes(pos(updated,1), :);
      end % trans_LR
   end   
end % level

for pi = 1:model.num_parts
   if(iscached(pi))
      continue;
   end

   hyp = update_hyp(hyp, best_scores{pi}, best_loc{pi}, best_box{pi}, pi);    
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



         
function [boxes scores g_hog_blocks num_blocks sz] = detect_poselets_sc(hog_st, pm, g_hog_blocks, num_blocks0, sz) % This isn't the most efficient way, but feature computation is already slooow


   % They encode size in [x y], we use [r c]
   if(isempty(g_hog_blocks) || any(sz~=[pm.size hog_st.scale]))
      %fprintf('Recomputing!\n');
      [g_hog_blocks num_blocks] = hog2features_local(hog_st.hog, (pm.size([2 1])+1)*8); % returns in g_hog_blocks
      sz = [pm.size hog_st.scale];
   else
       num_blocks = num_blocks0;
   end
       
   if(prod(num_blocks)==0)
      boxes = zeros(0, 4);
      scores = [];
      return;
   end


   samples_x = hog_st.samples_x;
   samples_y = hog_st.samples_y;


   [qx,qy] = meshgrid(samples_x(1:num_blocks(1)),samples_y(1:num_blocks(2)));

   scores = g_hog_blocks*pm.filter(:)+pm.bias; %repmat(svms.svms(end,:),prod(num_blocks),1);

   % Now compute the boxes!
   top_left = bsxfun(@plus, [qx(:) qy(:)]+1,  hog_st.img_top_left);
   bot_right = bsxfun(@plus, top_left, (pm.size(1:2)+1)*8);

   boxes = [top_left, bot_right]*hog_st.scale;
