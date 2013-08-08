function model = train_bbox(D, cached_scores, model, calib_params)

cls = model.cls;

[Dpos inds] = LMquery(D, 'object.name', cls, 'exact');
cached_pos = cached_scores(inds);

OV_THRESH = 0.35;

DO_REG = 1;

% Collect examples for each part
for i = 1:length(Dpos)
   if(isempty(cached_pos{i}.regions))
       continue;
   end
   
   bbox = LMobjectboundingbox(Dpos(i).annotation, cls);

   [best_overlap obj_ind] = max(bbox_overlap_mex(bbox, cached_pos{i}.regions), [], 1);
   ok_reg = best_overlap>=OV_THRESH;

   if(~any(ok_reg))
       continue;
   end
   
   obj_box = bbox(obj_ind(ok_reg), :);
   all_obj_boxes{i} = obj_box;
   for j = 1:model.num_parts+DO_REG
      rec_box = zeros(sum(ok_reg), 4);

      if(j==model.num_parts+1) % Use the region!
         part_box = cached_pos{i}.regions(ok_reg, :);
         flipped = false(sum(ok_reg), 1);
         probs = ones(sum(ok_reg), 1);
      else
         part_box = double(cached_pos{i}.part_boxes(ok_reg, 4*(j-1) + [1:4]));
         flipped = double(cached_pos{i}.part_trans(ok_reg, j))==2;
         probs = sigmoid(cached_pos{i}.part_scores(ok_reg, j), calib_params{j});

         if(~isempty(model.part(j).spat_const) && model.part(j).spat_const(5)>0) % This is going to screw things up...
            continue; 
         end
      end


      %diag = sqrt(sum((part_box(:, [3 4]) - part_box(:, [1 2])).^2, 2));
      diag = repmat(part_box(:, [3 4]) - part_box(:, [1 2]), [1 2]);
      cent = 1/2*(part_box(:, [3 4]) + part_box(:, [1 2]));

      %rec_box = bsxfun(@rdivide, (obj_box - [cent cent]), diag);
      rec_box = (obj_box - [cent cent])./diag;

      % Handle flipped case... (x is flipped, y stays the same)
      if(any(flipped))
         rec_box(flipped, [1, 3]) = -rec_box(flipped, [3 1]);%bsxfun(@rdivide, [cent(flipped, [1 1]) - obj_box(flipped, [1 3])], diag(flipped));
         %rec_box(flipped, [2  4]) = bsxfun(@rdivide, [cent(flipped, [2 2]) - obj_box(flipped, [2 4])], diag(flipped));
      end

      all_weights{i, j} = probs;
      all_targets{i, j} = rec_box;
   end
end

%%%%%%%%%%%%%%%%%%%%%%% Learn box offsets %%%%%%%%%%%%%%%%
%%% Start by suppressing redundant parts ....

all_targets_supp = cell(size(all_targets));
all_weights_supp = cell(size(all_weights));

for i = 1:numel(all_targets)
   inds = nms_iou(all_targets{i}, 0.99);
   all_targets_supp{i} = all_targets{i}(inds, :);
   all_weights_supp{i} = all_weights{i}(inds, :);
end

for j = 1:model.num_parts+DO_REG
   % Find least average estimate of box regression
   targets_j = cat(1, all_targets_supp{:, j});
   targets_j(isinf(targets_j) | isnan(targets_j)) = 0;
   weights_j = cat(1, all_weights_supp{:, j});

   [dk inds] = sort(weights_j, 'descend');
   weights_j(inds(max(ceil(end/10),2):end)) = 0; % only use the top 10% of the detections

   box_offsets{j} = weights_j'*targets_j/sum(weights_j); % Weighted average of boxes
end


% Repeat the process above, but predict box for each example using offsets

% Collect examples for each part
for i = 1:length(Dpos)
   if(isempty(cached_pos{i}.regions))
       continue;
   end
   
   bbox = LMobjectboundingbox(Dpos(i).annotation, cls);

   [best_overlap obj_ind] = max(bbox_overlap_mex(bbox, cached_pos{i}.regions), [], 1);
   ok_reg = best_overlap>=OV_THRESH;

   obj_box = bbox(obj_ind(ok_reg), :);
   all_obj_boxes{i} = obj_box;
   for j = 1:model.num_parts+DO_REG
      if(j==model.num_parts+1) % Use the region!
         part_box = cached_pos{i}.regions(ok_reg, :);
         flipped = false(sum(ok_reg), 1);
         %probs = ones(sum(ok_reg), 1);
      else
         part_box = double(cached_pos{i}.part_boxes(ok_reg, 4*(j-1) + [1:4]));
         flipped = double(cached_pos{i}.part_trans(ok_reg, j))==2;
         %probs = sigmoid(cached_pos{i}.part_scores(ok_reg, j), calib_params{j});
         if(~isempty(model.part(j).spat_const) && model.part(j).spat_const(5)>0) % This is going to screw things up...
            continue; 
         end
      end

      %diag = sqrt(sum((part_box(:, [3 4]) - part_box(:, [1 2])).^2, 2));
      diag = repmat(part_box(:, [3 4]) - part_box(:, [1 2]), [1 2]);
      cent = 1/2*(part_box(:, [3 4]) + part_box(:, [1 2]));
      
      pred_box0 = repmat(box_offsets{j}, size(cent,1), 1);
      pred_box0(flipped, :) = bsxfun(@times, pred_box0(flipped, [3 2 1 4]), [-1 1 -1 1]);
      
      %pred_box = bsxfun(@times, pred_box0, diag) + [cent cent];
      pred_box = pred_box0.*diag + [cent cent];

      all_pred{i, j} = pred_box;
      all_flipped{i, j} = flipped;
   end
end


% Organize training data
obj_ind = 0;
for i = 1:length(Dpos)
   for o = 1:size(all_obj_boxes{i},1)
      all_pred_t = zeros(size(all_pred,2), 6);
      for p = 1:size(all_pred,2)
         if(~isempty(all_pred{i,p}))
            all_pred_t(p, :) = [all_pred{i, p}(o, :) all_weights{i, p}(o) all_flipped{i, p}(o)];
         end
      end

      if(sum(all_pred_t(:, 5))>1e-5) % At least one ``confident'' part, store results
         obj_ind = obj_ind + 1;
         to_pred{obj_ind} = all_obj_boxes{i}(o,:);
         all_pred_combined{obj_ind} = all_pred_t;
      end
   end
end


all_pred_separate = {};
to_pred_mat = cat(1, to_pred{:});

diag = sqrt(sum((to_pred_mat(:, [3 4]) - to_pred_mat(:, [1 2])).^2, 2)); % This scales all GT boxes


Yw = bsxfun(@rdivide, to_pred_mat, diag);


Nfeat = model.num_parts + DO_REG;
A = kron(eye(4), ones(1, Nfeat));
b = ones(4, 1);


all_pred_weighted = {}; %all_pred_combined;
Ytodo = [];
ind = 0;
for i = 1:length(all_pred_combined)
   if(sum(all_pred_combined{i}(1:Nfeat, 6))>1e-4)
      ind = ind + 1;
      all_pred_weighted{ind} = all_pred_combined{i}(1:Nfeat, :);
      all_pred_weighted{ind}(1:Nfeat, 1:4) = all_pred_combined{i}(1:Nfeat, 1:4)/diag(i);
      Ytodo(ind, :) = Yw(i, :);
   end
end


%w0 = [eps+zeros(model.num_parts, 4); ones(DO_REG, 4)];
w0 = [ones(Nfeat-1, 4); ones(1, 4)];
%w0 = [zeros(Nfeat-1, 4); ones(1, 4)];

options = optimset;

opts.method = 'lbfgs';
opts.maxIter = 2000;
opts.verbose = 1;

lb = zeros(Nfeat*4, 1)+0.001;
ub = 100*ones(Nfeat*4, 1);

tic;
weights = minConf_TMP(@(w)fobj_bbox_weighting_flip(w, Ytodo, all_pred_weighted), w0(:), lb, ub, opts);
weights = reshape(weights, [], 4);

for i = 1:4
   params{i} = weights(:, i)/sum(weights(:, i));
end

model.box_model.box_offsets = box_offsets;
model.box_model.part_weights = params;
model.box_model.do_flip = 1;

reg_ind = find(strcmp(model.region_model, 'pred_box_overlap'));

if(isempty(reg_ind))
   reg_ind = length(model.region_model) + 1;
end

model.region_model{reg_ind} = 'pred_box_overlap';

return;

%weights = fmincon(@(w)fobj_bbox_weighting_flip(w, Yw, all_pred_weighted), w0(:), A, b, [],[], zeros(Nfeat*4, 1), [], [], options);

return;

for i = 1:4
   for obj_ind = 1:length(all_pred_combined)
      all_pred_separate{obj_ind, i} = [all_pred_combined{obj_ind}(:, i); all_pred_combined{obj_ind}(:, 5); all_pred_combined{obj_ind}(:, 6)];
   end

   Y = to_pred_mat(:, i);
   Yw = bsxfun(@rdivide, Y, diag);
   X = cat(2, all_pred_separate{:, i})';

   Xw = X;
   Xw(:, 1:end/3) = bsxfun(@rdivide, X(:, 1:end/3), diag);
keyboard
  % params{i} = nlinfit(Xw, Y./diag, @box_fit, [zeros(model.num_parts, 1); ones(DO_REG, 1)]); %1./(model.num_parts+DO_REG)*ones(model.num_parts+DO_REG, 1));
  params{i} = nlinfit(Xw, Yw(:), @box_fit_flip, [zeros(model.num_parts, 4); ones(DO_REG, 4)]); %1./(model.num_parts+DO_REG)*ones(model.num_parts+DO_REG, 1));
  params{i} = max(params{i}, 0);
  params{i} = params{i}/sum(params{i});
%   sig{i} = nlinfit(X, Y, @box_fit, ones(model.num_parts, 1));
end









