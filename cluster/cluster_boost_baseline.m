function cluster_test(cls, trainval, refinement_type)

% Parse input
if(~isinf(str2num(cls)))
   cls = str2num(cls);
end

try
   trainval = str2num(trainval);
end

%matlabpool_robust;

%%%%%%%%%%%%%%%%% Setup class... %%%%%%%%%%%%%%%%%%%%%%%%%%
startup_cluster
BDglobals;

VOCinit;
VOCopts.sbin = 8;
VOCopts.localdir = fullfile(WORKDIR, 'exemplars');

if(isnumeric(cls))
   clsind = cls;
   cls = VOCopts.classes{clsind};
end

fprintf('Doing category: %s\n', cls);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(trainval)
   set_str = 'trainval';
   load_init_final;
   load_init_test;
else % Just the training set
   set_str = 'train';
   load_init_data;
end


base_dir = fullfile('data/results', cls, sprintf('part_models_%s_%s', set_str, refinement_type));
det_str = fullfile(base_dir, sprintf('part_detections.mat'));
load(det_str, 'model', 'cached_scores', 'cached_scores_test');


result_str = fullfile(base_dir, sprintf('boost_model_baseline.mat'));


% THese are all done without LOO estimates

%methods = {'thresh_java', 'sigmoid_java', 'svm'};
methods = {'thresh_java', 'sigmoid_java', 'svm'};
Cs = 5.^[-1]

results_test = cell(1, length(methods));


[Dpos inds] = LMquery(D, 'object.name', cls, 'exact');
cached_pos_noloo = cached_scores(inds);
% Need to first compute NON-LOO
if(~exist(result_str, 'file'))
   [model.part.computed] = deal(0);
   [dk cached_pos_noloo] = collect_boost_data(model, Dpos, cached_pos_noloo);
   save(result_str, 'cached_pos_noloo'); % Save now in case it crashes!
else
   load(result_str ); % Load everything so we can pick up where we left off
end

methods = {'thresh_java', 'sigmoid_java', 'svm'};
cached_scores(inds) = cached_pos_noloo;

% Replace with relocalized windows
loc_str = fullfile(base_dir, sprintf('loc_model.mat'));
load(loc_str, 'model_wbox');

nobox = 1;

for i = 1:length(methods)
   method = methods{i};

   if(~isempty(results_test{i}))
      continue;
   end
if(nobox==1)
   fprintf('Boxes!\n');
   cached_scores = box_inference(cached_scores, model_wbox);
   cached_scores_test = box_inference(cached_scores_test, model_wbox);

   nobox = 0;
end
   switch method
      case 'svm'
         for Ci = 1:length(Cs)
            [learner{i}{Ci}] = boost_iterate(D, cached_scores, cls, 10, methods{i}, [], Cs(Ci));

            cached_scores_test = apply_weak_learner(cached_scores_test, learner{i}{Ci});

            results_test{i}{Ci} = test_voc(Dtest, cached_scores_test, cls, 0.5);
         end
      case {'thresh_java', 'sigmoid_java'}
         for Ci = 1:1

            [learner{i}{Ci}] = boost_iterate(D, cached_scores, cls, 3, methods{i});
            cached_scores = apply_weak_learner(cached_scores, learner{i}{Ci});
            cached_scores_test = apply_weak_learner(cached_scores_test, learner{i}{Ci});

            results_test{i}{Ci} = test_voc(Dtest, cached_scores_test, cls, 0.5);
         end
   end
end


save(result_str, 'cached_pos_noloo', 'results_test', 'learner', 'Cs', 'methods');





