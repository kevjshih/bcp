function [ex_im ex_bbox im_list im_name_list obj_box obj_box0 best_part_boxes best_scores reg_box] = get_exemplar_boxes(D, cached_scores, cls, ex_model, ind)
%(ex_im, ex_bbox, ims, obj_bboxes, det_bboxes, scores)

BDglobals;

if(exist('ex_model','var') && ~isfield(ex_model, 'part'))
   candidate_dir = fullfile(WORKDIR, 'candidates', cls);
   results = load(fullfile(candidate_dir, ex_model.name));

   [ex_im ex_bbox] = extract_exemplar_params(ex_model);
   ex_im = fullfile(im_dir, ex_im); 

   use_cached = 0;
elseif(exist('ex_model','var') && isfield(ex_model, 'part') && exist('ind','var'))
   [ex_im ex_bbox] = extract_exemplar_params(ex_model.part(ind));
    ex_im = fullfile(im_dir, ex_im); 
    
   use_cached = 1;
else
   error('Incorrect parameters\n');
end

[Dpos inds] = LMquery(D, 'object.name', cls, 'exact');
cached_pos = cached_scores(inds);

if(~use_cached)
    part_scores = results.part_scores(inds);
    part_det = results.part_detections(inds);
end

best_scores = {};
best_part_boxes = {};
im_list = {};
obj_box = {};
obj_box0 = {};
reg_box = {};
im_name_list = {};

for i = 1:length(Dpos)
   obj_inds = unique(cached_pos{i}.labels(cached_pos{i}.labels>0));

   bboxes = LMobjectboundingbox(Dpos(i).annotation, cls);
   for j = obj_inds(:)' 
      ok = find(cached_pos{i}.labels==j);

      if(use_cached)
         % Johnston: Use .scores(ok) instead of .part_scores(ok, ind)
         [best_scores{end+1} best_ind0] = max(cached_pos{i}.part_scores(ok, ind));
      else
         [best_scores{end+1} best_ind0] = max(part_scores{i}(ok));
      end

      best_ind = ok(best_ind0);
      if(use_cached)
         best_part_boxes{end+1} = cached_pos{i}.part_boxes(best_ind, (ind-1)*4 + (1:4));
      else
         best_part_boxes{end+1} = part_det{i}(best_ind,:);
      end

      if(isfield(ex_model, 'do_transform') && ex_model.do_transform==1)
         im = imread(fullfile(im_dir, Dpos(i).annotation.filename));

         if(cached_pos{i}.part_trans(best_ind, ind)==2) % Flip it!
            im_list{end+1} = im(:, end:-1:1, :);
            obj_box{end+1} = flip_box(bboxes(j,:), size(im));
            reg_box{end+1} = flip_box(cached_pos{i}.regions(best_ind, :), size(im));
            best_part_boxes{end} = flip_box(best_part_boxes{end}, size(im));
         else
            im_list{end+1} = im;
            obj_box{end+1} = bboxes(j, :);
            reg_box{end+1} = cached_pos{i}.regions(best_ind, :);
         end
      else
         im_list{end+1} = fullfile(im_dir, Dpos(i).annotation.filename);
         obj_box{end+1} = bboxes(j, :);
         reg_box{end+1} = cached_pos{i}.regions(best_ind, :);
      end
      
      im_name_list{end+1} = Dpos(i).annotation.filename;
      obj_box0{end+1}  = bboxes(j, :);
   end
end


% Sort by scores
[a b] = sort(cat(1,best_scores{:}), 'descend');
if(length(b)~=length(best_scores))
   error('Something went wrong!\n');
end

best_scores = best_scores(b);
best_part_boxes = best_part_boxes(b);
im_list = im_list(b);
im_name_list = im_name_list(b);
obj_box = obj_box(b);
reg_box = reg_box(b);
obj_box0 = obj_box0(b);


return;
% verify it:;



for i = 1:length(im_list)
   clf;
   im = imread(im_list{i});
   imagesc(im);
   hold on;
   draw_bbox(obj_box{i}, 'linewidth', 3);
   draw_bbox(reg_box{i}, '--r', 'linewidth', 2);
   draw_bbox(best_part_boxes{i}, 'g');
   pause;
end
