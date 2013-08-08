function [feat all_labels scores] = collect_multiclass(D, cached_scores, cls, model)

%error('Don''t want to do this anymore!!!!\n');
BDglobals;
VOCinit;

codebook = compute_codebook(D, 1024);

%categories =  VOCopts.classes;


if(~exist('model', 'var'))
   model = [];
end


%parfor_progress(length(D));
parfor i = 1:length(D)
%    parfor_progress();
    fprintf('%d/%d\n', i, length(D));
    [feat{i} all_labels{i}, scores{i}] = helper(D(i).annotation, cached_scores{i}, cls, codebook, model);
end

function [feat labels, scores] = helper(annotation, cached_scores, categories, codebook, model)
   feat = [];
   labels = [];
   scores = [];

   if(isempty(cached_scores.regions))
      return;
   end

   boxes = LMobjectboundingbox(annotation, categories);

   regions = [];
   to_check = [];
   if(isempty(boxes)) % Collect negatives
      if(isempty(model)) % Randomly sample negatives!
         r = randperm(size(cached_scores.regions,1));
         okneg = r(1:min(5, end)); % Start with some small number?

         regions = cached_scores.regions(okneg, :);
         labels = -ones(numel(okneg), 1);
      else
         to_check = 1:size(cached_scores.regions,1);
      end
   else
      if(isempty(model))
         %[ov best_reg] = max(bbox_overlap_mex(boxes, cached_scores.regions), [], 2);
         %ok = find(ov>0.5);
      
         labels = ones(size(boxes,1), 1); %find(strcmp(categories, names{ok(j)}));
         regions = boxes; %cached_scores.regions(best_reg(ok),:);
      else
         [ov best_gt] = max(bbox_overlap_mex(boxes, cached_scores.regions), [], 1);
         %ok_pos = find(ov>0.5);
         to_check = find(ov<0.2);
         %...
      end
   end


   if(~isempty(to_check))
      regions = cached_scores.regions(to_check,:);
   end

   if(~isempty(regions))
      feat = compute_bow_features(annotation, regions, codebook)';
   end

   if(~isempty(to_check)) % Now figure out which are the most violated
      d = svmpredict_approx(feat, model); % These will be used to train next layer
      violated = find(d>=-1.03);

      inds = nms_v4([regions(violated,:)  reshape(d(violated), [], 1)], 0.7);
      
      final_inds = violated(inds(1:min(end,5))); % Take at most 5 per image

      feat = feat(final_inds, :);
      labels = -ones(length(final_inds), 1);
      scores = d(final_inds);

   end

