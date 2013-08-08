function show_top_detections(boxes, D, direction)

BDglobals;

for i = 1:length(D)
   D(i).annotation.folder = '';
end

all_boxes = cat(1,boxes{:});
all_scores = all_boxes(:,end);


for i = 1:length(boxes)
   im_ind{i} = i*ones(size(boxes{i},1), 1);
end

all_inds = cat(1, im_ind{:});


if(~exist('direction', 'var') || direction == 1)
   [dk best_inds] = sort(all_scores, 'descend');
else
   [dk best_inds] = sort(all_scores, 'ascend');
end 

dims = [5 5];

for i_outer = 0:prod(dims):length(best_inds)
   clf
   for i = 1:prod(dims)
      subplot(dims(1), dims(2), i);

      im = LMimread(D, all_inds(best_inds(i+i_outer)), im_dir);
      imagesc(im);
      hold on;
      axis image off;
      draw_bbox(all_boxes(best_inds(i+i_outer),:));
   end
   pause
end
