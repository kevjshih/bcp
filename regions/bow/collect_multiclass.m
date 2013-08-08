function [feat all_labels] = collect_multiclass(D, cached_scores, model)

BDglobals;
VOCinit;

codebook = compute_codebook(D, 1024);

categories =  VOCopts.classes;


if(~exist('model', 'var'))
   model = [];
end


parfor_progress(length(D));
parfor i = 1:length(D)
    parfor_progress();
    fprintf('%d/%d\n', i, length(D));
    [feat{i} all_labels{i}] = helper(D(i).annotation, cached_scores{i}, categories, codebook, model);
end
parfor_progress(0);

function [feat labels] = helper(annotation, cached_scores, categories, codebook, model)
   feat = [];
   labels = [];

   if(isempty(cached_scores.regions))

      return;
   end

   boxes = LMobjectboundingbox(annotation);

   labels = [];
   if(~isempty(boxes))
      names = {annotation.object.name};

      [ov best_reg] = max(bbox_overlap_mex(boxes, cached_scores.regions), [], 2);

      ok = find(ov>0.5);
      
      for j = 1:length(ok(:))
         labels(j) = find(strcmp(categories, names{ok(j)}));
      end

      regions = cached_scores.regions(best_reg(ok),:);


      [ovneg best_ind] = max(bbox_overlap_mex(boxes, cached_scores.regions), [], 1);
      okneg = find(ovneg<=0.2);
      r = randperm(length(okneg));
      okneg = okneg(r(1:length(ok))); % Find equal proportions of positive and negative regions

      regions = [regions; cached_scores.regions(okneg, :)];
      labels = [labels(:); -ones(length(okneg), 1)];
   
      fprintf('%d: %d\n', sum(labels>0), length(labels));
      
      feat = compute_bow_features(annotation, regions, codebook);
   end
