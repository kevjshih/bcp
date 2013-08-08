function [labels_sub cached_sub imind] = update_boost_set(labels1, cached1, imind1, labels2, cached2, imind2, update_pos, classifier)

if(~exist('classifier', 'var'))
   classifier = [];
end

% Find duplicates between old set and new set
neg_set1 = cached1(labels1<=0,:);
neg_im1 = imind1(labels1<=0);
neg_set1(isinf(neg_set1)) = -100; % Don't let infinities screw things up

new_neg = find(labels2<=0);
neg_im2 = imind2(labels2<=0);
neg_set2 = cached2(labels2<=0, :);
neg_set2(isinf(neg_set2)) = -100;

dup = zeros(numel(new_neg), 1);

for i = unique([neg_im1(:); neg_im2(:)])'
   todo1 = neg_im1==i;
   todo2 = neg_im2==i;

   dup(todo2) = find_duplicates_row(neg_set1(todo1, :), neg_set2(todo2, :));
end

if(update_pos)
   cached_pos = [cached2(labels2==1,:)]; % Use all of the new positive examples
   imind_pos = imind2(labels2==1);
else
   cached_pos = [cached1(labels1==1,:)]; % Use all of the new positive examples
   imind_pos = imind1(labels1==1);
end

cached_neg = [cached1(labels1<=0, :); cached2(new_neg(~dup),:)];
im_ind_neg = [imind1(labels1<=0); imind2(new_neg(~dup))];

if(~isempty(classifier)) % Prune negatives to a managable size ....
   pred = boost_classify(cached_neg, classifier);
   pred_sort = sort(pred, 'descend');
   th = pred_sort(min(end, 200000)); % 75000, keep everything for now
%   th = pred_sort(min(end, inf)); % 75000, keep everything for now

   keep = pred>-inf;
   %keep = pred>=th;

   cached_neg = cached_neg(keep,:);
   im_ind_neg = im_ind_neg(keep);
end

%cached_pos = [cached1(labels1==1,:)]; % Use all of the new positive examples
%imind = [imind1(labels1==1); im_ind_neg];

imind = [imind_pos; im_ind_neg];
cached_sub = [cached_pos; cached_neg];
labels_sub = [ones(size(cached_pos,1),1); -ones(size(cached_neg,1),1)];
