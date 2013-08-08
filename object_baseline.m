javaaddpath('/home/engr/iendres2/prog/tools/JavaBoost/dist/JBoost.jar');

cls = 'cat';

load_init_data;

% Cluster examples based on aspect ratio
Nsplits = 3;

[dk inds] = LMquery(D, 'object.name', cls, 'exact');
Dneg = D;
Dneg(inds) = [];
cached_scores_neg = cached_scores;
cached_scores_neg(inds) = [];

[Dpos_split Dpos_inds cached_split aspect] = cluster_pos(D, cls, cached_scores, Nsplits);

% Train all of the models
empty_model = model;
for i = 1:Nsplits
   % Init appearance model
   split_model{i} = add_model(empty_model, init_clustered_model(aspect(i)));

   % Initialize it using only the best region locations
   split_model{i}.do_loo = 0;
   split_model{i} = train_loo(split_model{i}, [Dpos_split{i} Dneg] , [cached_split{i} cached_scores_neg], 10, 2, 1);

   % Now let it choose from any region
   split_model{i}.do_loo = 1;
   [split_model{i} loo_split{i}= train_loo(split_model{i}, [Dpos_split{i} Dneg] , [cached_scores(Dpos_inds{i}) cached_scores_neg], 7, 4, 1);
end
