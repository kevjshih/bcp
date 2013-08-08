function[neg_inds_all] = mine_negatives_for_cascade(D, cached_scores, target_quantity,...
                                                  cls)
% random sampling for negative examples 
disp(['mining ', num2str(target_quantity), ' negatives...']);

neg_inds_all = cell(length(cached_scores),1);

neg_inds_raw = [];
neg_im_raw = [];
disp('finding available negatives...');
for i = 1:length(cached_scores)
    neg_inds = find(cached_scores{i}.labels < 0);
    neg_inds_raw = [neg_inds_raw; neg_inds];
    neg_im_raw = [neg_im_raw; ones(length(neg_inds),1).*i];   
end

num_negs = length(neg_inds_raw);

rand_order = randperm(num_negs);
if num_negs > target_quantity
    rand_order = rand_order(1:target_quantity);
end
disp('randomly selecting negatives...');
ordered_picks = sort(rand_order);
neg_inds_picked = neg_inds_raw(ordered_picks);
neg_im_picked =  neg_im_raw(ordered_picks);

picked_ims = unique(neg_im_picked);

for i = picked_ims'
   neg_inds_all{i} = neg_inds_picked(neg_im_picked == i); 
end
disp(['succefully mined ', num2str(length(rand_order)), ' negatives']);