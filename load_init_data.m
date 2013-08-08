if(~exist('cls', 'var'))
   error('You need to define cls!');
end
% This version trains everything on train set

basedir = fileparts(which('load_init_data.m'))

BDglobals;
BDpascal_init;

init_dir = fullfile(basedir, 'data', 'initdata');
init_data = fullfile(init_dir, [cls, '_', VOCyear, '_init.mat']);

if(exist(init_data, 'file'))
   start1 = tic;
   load(init_data, 'D', 'Dtest', 'model','cached_scores', 'cached_scores_test');
   stop1 = toc(start1);
   start2 = tic;
   fprintf('Loaded initial %s data: %d, updated path: %d\n', cls, stop1, toc(start2));

   if(0&&~any(strcmp(model.region_model, 'Area model')))
      % Adding two models for area and aspect ratio
      model.region_model{end+1} = 'Area model';
      model.region_model{end+1} = 'Area model';
      
      cached_scores = add_region_area(model, D, cached_scores);
      cached_scores_test = add_region_area(model, Dtest, cached_scores_test);
      save(init_data);
   end
else
   if(~exist(init_dir, 'file'))
      mkdir(init_dir);
   end

   D = pasc2D('train', VOCopts);
   Dtest = pasc2D('val', VOCopts);

   model = init_model(cls);

   D = update_D_categories(D, cls); % If cls isn't one of the basic level categories, replace it as needed
   Dtest = update_D_categories(Dtest, cls); % If cls isn't one of the basic level categories, replace it as needed
   
   cached_scores = init_cached_scores(model, D);
   cached_scores_test = init_cached_scores(model, Dtest);

   model = train_region_model(D, cached_scores, model);
   cached_scores = add_region_scores(model, D, cached_scores);
   cached_scores_test = add_region_scores(model, Dtest, cached_scores_test);

   whole_path = path;
   save(init_data);
end
