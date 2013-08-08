function train_candidate_parts(cls, num, suffix, trainset)

BDglobals;

if(~exist('trainset', 'var'))
   trainset = 0; % Train on validation
end

if(~exist('num','var'))
   num = 2000;
end

if(~exist('suffix', 'var'))
   suffix = '';
end

if(~isempty(suffix))
   suffix = ['_' suffix];
end

if(trainset==0)
   trainset = 'train';
else
   trainset = 'trainval';
end

suffix = [suffix '_' trainset];

BDVOCinit;
VOCopts.sbin = 8;
VOCopts.localdir = fullfile(WORKDIR, ['exemplars' suffix]);

if(isnumeric(cls))
   clsind = cls;
   cls = VOCopts.classes{clsind};
   fprintf('Doing category: %s\n', cls);
end

VOCopts.trainset = TRAINSET;
fprintf('Using the training set: "%s"\n', VOCopts.trainset);

ps = get_pos_stream(VOCopts, VOCopts.trainset, cls);


if(num>0)
   [I bbox] = auto_get_part_fast(VOCopts, ps, num);
   models = orig_train_elda(VOCopts, I, bbox, cls, VOCopts.trainset, 0);
end

[I bbox] = auto_get_obj_part(VOCopts, ps);
models = orig_train_elda(VOCopts, I, bbox, cls, VOCopts.trainset, 0, 1);

% Also create filters at lower resolution:
for i = 1:length(bbox)
   bbox{i}(:,5) = 7;
end
models = orig_train_elda(VOCopts, I, bbox, cls, VOCopts.trainset, 0, 1);
%else
%end

