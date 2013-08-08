

datafile = 'data/initdata/mc_init.mat';

if(exist(datafile, 'var'))
   t = load(datafile);
else
   codebook = compute_codebook(D, 1024); 
   models = load('data/bow/tmp_bow_models.mat', 'model');


   bow_scores = apply_bow_all(D, cached_scores, codebook, models.model);
   bow_scores_test = apply_bow_all(Dtest, cached_scores_test, codebook, models.model);

   save(datafile, 'bow_scores', 'bow_scores_test');
   t.bow_scores = bow_scores;
   t.bow_scores_test = bow_scores_test;
end


for i = 1:length(cached_scores)
   cached_scores{i}.region_scores = [cached_scores{i}.region_scores t.bow_scores{i}];
end

for i = 1:length(cached_scores_test)
   cached_scores_test{i}.region_scores = [cached_scores_test{i}.region_scores t.bow_scores_test{i}];
end

clear t bow_scores bow_scores_test;


