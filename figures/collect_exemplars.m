function mosaic = collect_exemplars(D, cached_scores, cls, model, part_ind, method)

BDglobals;

NIPS_VERSION = 1;

if(NIPS_VERSION)

res_dir = fullfile(FIGDIR, cls, method, [num2str(part_ind) '_' model.part(part_ind).name]);
avg_dir = fullfile(FIGDIR, cls, method);
mkdir(res_dir);
mkdir(avg_dir);

if(0)
   h = figure(1);
   clf;
   box = ex_bbox;
   imagesc(imread(ex_im));
    hold on;
   draw_bbox(box, 'linewidth', 3);
   fname = fullfile(res_dir, sprintf('exemplar.png'));

   saveas_tight(h, fname);
else
      [ex_im ex_bbox] = extract_exemplar_params(model.part(part_ind).name);
      box = double(ex_bbox);%best_part_boxes{i});
      im = imread(fullfile(im_dir, ex_im));
      im = im(box(2):box(4), box(1):box(3), :);
      %im
      fname = fullfile(res_dir, sprintf('exemplar.png'));
      imwrite(im, fname);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GET TOP DETECTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(D)
   if(~isempty(cached_scores{i}.scores))
      all_detections = [double(cached_scores{i}.part_boxes(:,4*(part_ind-1) + [1:4])), cached_scores{i}.part_scores(:,part_ind)];
      det_ind = nms_v4(all_detections, 0.2);
      detections = all_detections(det_ind, :);
      boxes{i} = detections(:,1:4);
      scores{i} = detections(:, end);
      detection_list{i} = detections;
      inds{i} = repmat(i, length(det_ind), 1);
      flipped{i} = cached_scores{i}.part_trans(det_ind, part_ind); 
   else
      boxes{i} = zeros(0, 4);
      scores{i} = zeros(0,1);
   end

   [dk ids{i}] = fileparts(D(i).annotation.filename);
end

all_inds = cat(1, inds{:});
all_scores = cat(1, scores{:});
all_boxes = cat(1, boxes{:});
all_flipped = cat(1, flipped{:});


[dk best_inds] = sort(all_scores, 'descend');


% Write top 10 detections
interval = 1;

sz = model.part(part_ind).size(1:2)*8;

dims = [3 5];
pad = 2
mosaic = ones((sz(1)+pad)*dims(1), (sz(2)+pad)*dims(2), 3);

avg_im = zeros([sz*2 3]);
avg_grad = zeros([sz*2]);

Ntodo = 15;
for j = 1:Ntodo
      best_ind = best_inds(j);
      box = double(all_boxes(best_ind, :));
      im = im2double(imread(fullfile(im_dir, D(all_inds(best_ind)).annotation.filename)));
      im = padarray(im, [500 500 0]); % Just to be safe
      box = box + 500;
      im = im(box(2):box(4), box(1):box(3), :);

      if(all_flipped(best_ind)==2)
         im = im(:, end:-1:1, :);
      end
      %im

      im = imresize(im, sz*2); % want higher resolution here!

      avg_grad = get_grad2(im)/Ntodo + avg_grad;       
      avg_im = avg_im + im/Ntodo;
end


for j = 1:interval:15*interval
   if(0)
      clf;
      h = figure(1);
      % Crop image
      box = round(obj_box{i});
      box_sz = box([3 4]) - box([1 2]);
      box = box + [-box_sz/3 box_sz/3];
      imsz = size(im_list{i});
      box = max(1, min(imsz([2 1 2 1]), box));
      im = im_list{j}(box(2):box(4), box(1):box(3), :);
      imagesc(im);
      hold on;
      axis image
      axis off
      draw_bbox(double(best_part_boxes{j}) - [box(1) box(2) box(1) box(2)], 'linewidth', 3);
      fname = fullfile(res_dir, sprintf('%d.png', i));

      saveas_tight(h,fname);
   else
      best_ind = best_inds(j);
      box = double(all_boxes(best_ind, :));
      im = im2double(imread(fullfile(im_dir, D(all_inds(best_ind)).annotation.filename)));
      im = padarray(im, [500 500 0]); % Just to be safe
      box = box + 500;
      im = im(box(2):box(4), box(1):box(3), :);

      if(all_flipped(best_ind)==2)
         im = im(:, end:-1:1, :);
      end
      %im
      fname = fullfile(res_dir, sprintf('%d.png', j));
      imwrite(im, fname);

      ind = ceil(j/interval);
   
      [y x] = ind2sub(dims, ind);

      mosaic((y-1)*(sz(1)+pad) + [1:sz(1)], (x-1)*(sz(2)+pad) + [1:sz(2)], :) = imresize(im, sz);
   end
end
      fname = fullfile(avg_dir, sprintf('mosaic_%d.png', part_ind));
      imwrite(mosaic, fname);

      fname = fullfile(avg_dir, sprintf('average_%d.png', part_ind));
      imwrite(avg_im, fname);

      fname = fullfile(avg_dir, sprintf('average_grad_%d.png', part_ind));
      imwrite(avg_grad, fname);
else

   top_detections = collect_detections(cached_scores, i);
  
   for im = 1:length(top_detections)
      im_ind{im} = repmat(im, size(top_detections{im},1), 1);
   end

   all_detections = cat(1, top_detections{:});
   all_imind = cat(1, im_ind{:});

   [score best_ind] = sort(all_detections(:, end), 'descend');
   

   avg_im = zeros([8*model.part(i).size(1:2) 3]);

 
   for j = 1:10
      res_ind = bestind(j);

      box = double(all_detections(res_ind, 1:4));
      im = imread(imdir, D(i).annotation.filename);
      im = padarray(im, [500 500 0]); % Just to be safe
      box = box + 500;
      im = im(box(2):box(4), box(1):box(3), :);
      %im
      fname = fullfile(res_dir, sprintf('%d.png', j));
      imwrite(im, fname);

      avg_im = avg_im + imresize(im, [size(avg_im,1), size(avg_im,2)]);
   end

end
