%function collect_all_exemplars

classes = {'aeroplane', 'cat', 'person'};


for i = 1:length(classes)
   cls = classes{i};

   load_init_data;
   load(fullfile('data/results/', cls, 'train_analysis_final.mat'));

   % Only using positive data - NOT ANYMORE
   model.do_boxes = 1;
   [model.part.computed] = deal(0);

   model.part(11:end) = [];
   model.num_parts = 10;


%   [dk inds] = LMquery(Dtest, 'object.name', cls);
%   Dtest = Dtest(inds);
   
%   cached_scores_test = cached_scores_test(inds);
   [dk cached_scores_test] = collect_boost_data(model, Dtest, cached_scores_test); % Need to fill in the boxes

   for j = 1:10 % Top 10 parts
      collect_exemplars(Dtest, cached_scores_test, cls, model, j);
   end
end
