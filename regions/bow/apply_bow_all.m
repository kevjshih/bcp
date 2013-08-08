function [bow_scores classification] = apply_bow_all(D, cached_scores, codebook, models, classifier)
% m = load('data/bow/tmp_bow_models.mat');
% c = load('data/bow/tmp_bow_mc_model.mat');
% codebook = compute_codebook;
% [bow_scores classification] = apply_bow_all(D, cached_scores, codebook, m.model, c.mc_model);
im_dir = [];
BDglobals;

bow_scores = cell(numel(D), 1);


if(isempty(codebook))
   codebook = compute_codebook;
end

if(~exist('classifier', 'var'))
   classifier = [];
end

parfor i = 1:length(cached_scores)
   tic
   fprintf('%d/%d\n', i, length(cached_scores));

   if(~isempty(cached_scores{i}.regions))
      feat = compute_bow_features(D(i).annotation, cached_scores{i}.regions, codebook);

      bow_scores{i} = zeros(size(cached_scores{i}.regions, 1), length(models));

      for m = 1:length(models);
         bow_scores{i}(:, m) = svmpredict_approx(feat', models{m});
      end
      %bow_scores{i} = compute_features(im, cached_scores{i}.regions, codebook);
      if(~isempty(classifier))
         classification{i} = linpredict(zeros(size(bow_scores{i},1), 1), sparse(bow_scores{i}), classifier);
      end
   end
   toc;
end
