function [all_scores detection_loc flipped] = part_inference(input, part_model, regions)
% [scores detection_loc] = inference(input, part_model, regions)
% **input: image to run detector on
% **part_model: 
% Model should have at least the following fields
%        hg_size: [8 10 31]
%              w: [8x10x31 double]
%              b: 1.6677
% **regions: a R x 4 matrix encoding locations of region proposals
%    each row should encode [xmin, ymin, xmax, ymax]
%
%
% Output:  Each variable is a cell array for each of the M parts
% scores: 1 score for each region gives the highest scoring location of part m in the region
% detection_loc:  Rx4 the highest scoring location of part m for each region
% flipped: R x 1, 1 indicates that part m has been flipped for each region


% This defines the overlap requirements:
%    Let P be Part area, R be region area and
%    let I be intersection of P and R, and U be their union
% const = [min I/R, max I/R, min I/P, max I/P, min I/U, max I/U]
const = [0 1 0.8 1 0.005 1]; % By default, this says that the part must be 80% within the region and shouldn't be too small by requiring I/U be at least 0.5%

Nreg = size(regions,1);

sbin = 8;
interval = 10;

padx = 6;%ceil(part_model.hg_size(2)/2+1);
pady = 6;%ceil(part_model.hg_size(1)/2+1);

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

if(~iscell(part_model))
   part_model = {part_model};
end

for m = 1:length(part_model)
   best_scores = -inf(size(regions,1), 1);
   best_loc = ones(size(regions,1), 4); % [x y s f];
   for level = 1:length(feat)
      featr = feat{level};
      scale = sbin/scales(level);

      for trans_LR = 1:2
         if(trans_LR==1) % Standard
            filter = {part_model{m}.w};
         else
            filter = {flipfeat(part_model{m}.w)};
         end

         % Compute scores
         rootmatch_cell = fconv(featr(:,:,1:31), filter, 1, 1);
         rootmatch = rootmatch_cell{1};
         Sx = size(rootmatch,2);
         Sy = size(rootmatch,1);
         xs = xs0(1:Sy, 1:Sx);
         ys = ys0(1:Sy, 1:Sx);
    
         % Prune candidates: 
         min_score = min(best_scores);
         ok_score = rootmatch>min_score; % No need to consider these...
         xsub = xs(ok_score);
         ysub = ys(ok_score);
         boxes = rootbox(xsub, ysub, scale, padx, pady, part_model{m}.hg_size(1:2));
         scores = rootmatch(ok_score);

         %[best_score pos] = get_best_part(regions, boxes, scores, [], [], 0.8, [], 0.1);
         %[best_score pos] = get_best_part(regions, boxes, scores, [], [], 0.0001, [], 0.001); % Everything is ok
         [best_score pos] = get_best_part(regions, boxes, scores, const(1), const(2), const(3), const(4), const(5), const(6)); %0, 1, 0.8, 1, 0.01, 1);

         [best_scores ind] = max([best_scores, best_score], [], 2);
         updated = ind==2;
         best_loc(updated, :) = [xsub(pos(updated)), ysub(pos(updated)), repmat([level trans_LR], sum(updated), 1)];
      end
   end
   all_scores{m} = best_scores;
   detection_loc{m} = rootbox(best_loc(:, 1), best_loc(:,2), sbin./scales(best_loc(:,3)), padx, pady, part_model{m}.hg_size(1:2));
   flipped{m} = best_loc(:, 4) == 2;
end


function boxes = rootbox(x, y, scale, padx, pady, rsize)
x1 = (x(:)-padx).*scale(:)+1;
y1 = (y(:)-pady).*scale(:)+1;
x2 = x1 + rsize(2).*scale(:) - 1;
y2 = y1 + rsize(1).*scale(:) - 1;

boxes = [x1 y1 x2 y2];

function ok = check_overlap(boxes, regions)

ok = bbox_overlap_mex(boxes, regions, 0.25)>0.25;
