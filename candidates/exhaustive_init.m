function [xs_all ys_all levels_all all_boxes f scale] = exhaustive_init(im, box)


[f scale] = IEfeatpyramid(im, 8, 3);

[xs0 ys0] = meshgrid(1:size(f{1},2), 1:size(f{1},1));

min_side = 5;
cell_step = 2;
xs_all = {};
ys_all = {};
levels_all = {};
all_boxes = {};
% Sample all windows that cover 5%+ of the object
for i = 1:length(f)
   Sx = size(f{i},2);
   Sy = size(f{i},1);
   xs = xs0(1:Sy, 1:Sx);
   ys = ys0(1:Sy, 1:Sx);

   bx = rootbox_simp(xs(:), ys(:), 8/scale(i), 0, 0, [1, 1]);
   bx_x = rootbox_simp(1:Sx, ones(1, Sx), 8/scale(i), 0, 0, [1, 1]);
   bx_y = rootbox_simp(ones(1, Sy), 1:Sy, 8/scale(i), 0, 0, [1, 1]);

   ok = reshape(bbox_overlap_mex(bx, box)>0, Sy, Sx);

   [iny inx] = find(ok);

   rx = max(1, min(inx(:))-1):cell_step:min(max(inx(:))+1, Sx);
   ry = max(1, min(iny(:))-1):cell_step:min(max(iny(:))+1, Sy);

   % enumerate possible sides
   xside = [];
   yside = [];
   for y = ry
      for x = rx
         if(ry(end)-y+1 >=10)
            for w = min_side:min((rx(end)-x+1), 10)
               h = 10;
               xside(end+1,:) = [x x+w-1];
               yside(end+1,:) = [y y+h-1];
            end
         end

         if(rx(end)-x+1 >=10)
            for h = min_side:min((ry(end)-y+1), 9)
               w = 10;
               xside(end+1,:) = [x x+w-1];
               yside(end+1,:) = [y y+h-1];
            end
         end
      end
   end
   
   if(numel(yside)==0 || numel(xside)==0)
      continue;
   end
   % For each pair, 
%   [a b] = meshgrid(1:size(xside,1), 1:size(yside, 1));
   a = 1:size(xside,1);
   b = 1:size(yside,1);

   boxes = [bx_x(xside(a(:), 1), 1), bx_y(yside(b(:), 1), 2), bx_x(xside(a(:), 2), 3), bx_y(yside(b(:), 2), 4)];
   ok = bbox_contained(box, boxes)>0.05 & bbox_contained(boxes, box)'>0.8;

   all_boxes{i} = boxes(ok, :);
   xs_all{i} = xside(a(ok), :);
   ys_all{i} = yside(b(ok), :);
   levels_all{i} = repmat(i, sum(ok), 1);
end

xs_all = cat(1, xs_all{:});
ys_all = cat(1, ys_all{:});
levels_all = cat(1, levels_all{:});
all_boxes = cat(1, all_boxes{:});

