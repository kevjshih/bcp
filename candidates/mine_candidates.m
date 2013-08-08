function mine_candidates(cls, num)

try
% Initialize RNG to get different streams
% Adjust this line accordingly for repeatable experiments
RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)));

matlabpool_robust;
load_init_data;

VOCinit;
VOCopts.sbin = 8;

ps = get_pos_stream(VOCopts, VOCopts.trainset, cls);

[I bbox] = auto_get_part(VOCopts, ps, num);
% Automatically select exemplars
models = orig_train_exemplar(VOCopts, I, bbox, cls, VOCopts.trainset, 0);
if(~iscell(models))
   models = {models};
end

for i = 1:length(models)
   m = models{i};
   models{i} = m.model;
   models{i}.name = [m.models_name '.mat'];
end

test_candidate_detections(D, cached_scores, cls, models);
catch
fprintf('ERROR!!\n');
end

exit;
