load_init_final;

load(fullfile('data/results/', cls, 'boost_trainval_incremental_final.mat'));

cached_scores = apply_weak_learner(cached_scores, model.learner);

%roc = test_given_cache(D, 
[roc res] = test_given_cache(D, cached_scores, cls, 0.5);
th = roc.conf(max(find(roc.p>0.1)));


[Dsub inds] = LMquery(D, 'object.name', cls, 'exact');
show_detection_detail(D(inds), cached_scores(inds), model, th);
