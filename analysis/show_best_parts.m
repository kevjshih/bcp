% 



[dk inds] = LMquery(D, 'object.name', cls, 'exact');
Dpos = D(inds);
cached_pos = cached_scores(inds);

tmp_model = model;
[tmp_model.part.computed] = deal(0);

%[dk hyp] = collect_positives(tmp_model, Dpos, cached_pos);
[scores cached_data] = collect_boost_data(tmp_model, Dpos, cached_pos);


for part_type = 1:10
for i = 1:length(cached_data)

   % Find highest scoring part from valid window
   ok_pos = find(cached_data{i}.labels==1);
   if(isempty(ok_pos))
      best_score = -inf;
   else
      [best_score d] = max(cached_data{i}.part_scores(ok_pos, part_type)); % Using the first part for now
   end

   best_scores(i) = best_score;


   if(isinf(best_score))
      continue;
   end
   if(DISPLAY_PER_EXAMPLE)
   im = imread(fullfile(im_dir, Dpos(i).annotation.filename));%, im_dir);
      best_ind = ok_pos(d);
      clf;
      imagesc(im);
      hold on;
      draw_bbox(cached_data{i}.regions(best_ind,:), 'r', 'linewidth', 4);
      draw_bbox(cached_data{i}.part_boxes(best_ind, 4*(part_type-1) + [1:4]), 'b', 'linewidth', 4);
   end
end

[top_scores top_dets] = sort(best_scores, 'descend');

figure(part_type);
for rank_ind = 1:25%top_dets
   subplot(5,5,rank_ind);
  
   i = top_dets(rank_ind); 
   im = imread(fullfile(im_dir, Dpos(i).annotation.filename));%, im_dir);

   ok_pos = find(cached_data{i}.labels==1);
   if(isempty(ok_pos))
      best_score = -inf;
   else
      [best_score d] = max(cached_data{i}.part_scores(ok_pos, part_type)); % Using the first part for now
   end

   
      best_ind = ok_pos(d);

   part_box = cached_data{i}.part_boxes(best_ind, 4*(part_type-1) + [1:4]);

   part_im = im(max(1, part_box(2)):min(part_box(4), end), max(1,part_box(1)):min(part_box(3),end), :);
   imagesc(part_im);
   axis off
   axis image
   title(sprintf('%f',  best_score));
end
end
