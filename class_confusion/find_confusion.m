function cc_model = find_confusion(model, cls, D, cached_scores, ind)
%function cc_model = find_confusion(model, cls, D, cached_scores, ind)
% For a given list of part scores, find which categories are most commonly confused
VOCinit; % For class labels
BDglobals;

classes = VOCopts.classes;

% First find the point with 25% precision
[dk dk dk dk roc] = test_part_detections_D(cls, D, cached_scores, ind); % The part scores should be LOO estimates

thresh_ind = max(find(roc.p(1:ceil(end/2))>=0.25)); % Use half the list to avoid the weird evaluation bug....
thresh = roc.conf(thresh_ind);

[model.part.computed] = deal(1);
model.part(ind).computed = 0; % Only use features for one part
model.thresh = thresh;
% Find all GT boxes with a part score greater than ``thresh'' and extract features
feat = {};
labels = {};

%loop over every example
for i = 1:length(D)
%for i = 1:300
  fprintf('%d\n', i);
  feat{i} = [];
  labels{i} = [];
  
  % if there was not part scores calculated for this example
  if(isempty(cached_scores{i}.part_scores))
    continue;
  end
  ok_ind = cached_scores{i}.part_scores(:,ind)>=thresh;
  
  if(any(ok_ind)) % if any regions scored highly on ind-th part-detector
    gt_boxes = LMobjectboundingbox(D(i).annotation); % extract gt_boxes that i-th example have
    
    ovs = max(bbox_overlap_mex(gt_boxes, cached_scores{i}.regions(ok_ind,:)), [], 2);
    % the maximum overlap of the gt_boxes by any regions that pass thresh
    
    ok_gt = ovs>=0.5; % If a region with high scoring parts has 50% overlap with GT, compute features
    
    if(any(ok_gt))
      fprintf('%d: Extracting %d examples\n', i, sum(ok_gt));
      im = imread(fullfile(im_dir, D(i).annotation.filename));
      [hyp feat_data] = inference(im, model, gt_boxes(ok_gt, :));
      % hyp might have fewer bboxes than gt_boxes(ok_gt), since all high overlapping regions with gt might turn out to have
      % low response from the part detector
      
      ok_hyp = [hyp.region]; % reprune since gt overlap is more strict than latent region overlap
      color_pyramid_max_lvl = 4;
      feat{i} = hyp_to_auxfeat(model, hyp, feat_data, im, ind, color_pyramind_max_lvl);
      %feat{i} = hyp_to_feat(model, hyp, feat_data);
      
      ok_gt = find(ok_gt);
      labels{i} = get_class_ind(classes, {D(i).annotation.object(ok_gt(ok_hyp)).name});
      imind{i} = repmat(i, length(ok_hyp), 1);
    end
  end
end

all_feat = cat(2, feat{:}); 
all_labels = cat(1, labels{:});
all_imind = cat(1, imind{:});

%save('all_feat_labels_imind.mat','all_feat','all_labels','all_imind');

pos_cls_ind = get_class_ind(classes, {cls});

% Compute confusion:
confusions = hist(all_labels, 1:length(classes));
gt_det = confusions(pos_cls_ind);
confusions(pos_cls_ind) = 0;

% Find 3 most confused categories:
[num_confusions worst_classes] = sort(confusions, 'descend');

to_use = worst_classes(find(num_confusions(1:3)>gt_det*0.1)); % Need to be least 10% as many confusions as positives

Cs = 5.^[-3:2];%svm cost constants


if(any(to_use))
  fprintf('Worst Classes: %s\n', classes{to_use});
  
  for i = 1:length(to_use)
    % Train the loo svm!!
    conf_cls = to_use(i);
    cur_labels = zeros(size(all_labels));
    cur_labels(all_labels==pos_cls_ind) = 1;
    cur_labels(all_labels==conf_cls) = -1;
    
    feat_sub = all_feat(:, cur_labels~=0);
    labels_sub = cur_labels(cur_labels~=0);
    imind_sub = all_imind(cur_labels~=0);
    
    cc_model(i).part_ind = ind;
    [cc_model(i).w_noloo cc_model(i).w_loo] = train_gen_loo(labels_sub, feat_sub, Cs, imind_sub);
    if(length(cc_model(i).w_loo)<length(D))
      w_loo{length(D)} = [];
    end
    
    cc_model(i).imind = imind_sub;
    cc_model(i).conf_class = classes(to_use(i));
    cc_model(i).conf_class_ind = to_use(i);
    cc_model(i).thresh = thresh;
  end
end

%function [feat labels] = mine_examples(cls

function inds = get_class_ind(all_classes, to_check)
[cls_list dk inds] = unique([all_classes(:); to_check(:)]);
if(length(cls_list)~=length(all_classes))
  error('Something went wrong!\n');
end

inds = inds(length(all_classes)+1:end);
