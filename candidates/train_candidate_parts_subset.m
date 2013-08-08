function train_candidate_parts_subset(cls, trial, numpos, dpm_pos)

% trial keeps track of the current trial id

BDglobals;

% mine 10 parts per positive example

parts_per_ex = 10;
numpos_examples = numpos;
num = min(2000, numpos_examples*parts_per_ex);
% num : number of parts to be mined total



VOCinit;
VOCopts.sbin = 8;
VOCopts.parts_per_ex = parts_per_ex;
%VOCopts.localdir = fullfile(WORKDIR, 'exemplars');
VOCopts.localdir = fullfile(WORKDIR, 'subset_experiments', cls, ...
                             'exemplars', num2str(numpos), num2str(trial));

if(isnumeric(cls))
   clsind = cls;
   cls = VOCopts.classes{clsind};
   fprintf('Doing category: %s\n', cls);
end

VOCopts.trainset = TRAINSET;
fprintf('Using the training set: "%s"\n', VOCopts.trainset);

%ps = get_pos_stream(VOCopts, VOCopts.trainset, cls);
% transform dpm_pos to get_pos_stream form

% randomly sample for subset of positives
%ps = ps{randsample(length(ps), numpos)};
ps = dpm_pos;

if(num>0)
   [I bbox] = auto_get_part_fast_subset(VOCopts, ps, num);
   models = orig_train_elda(VOCopts, I, bbox, cls, VOCopts.trainset, 0);
end

[I bbox] = auto_get_obj_part(VOCopts, ps);
models = orig_train_elda(VOCopts, I, bbox, cls, VOCopts.trainset, 0, 1);

% Also create filters at lower resolution:
for i = 1:length(bbox)
   bbox{i}(:,5) = 6;
end
models = orig_train_elda(VOCopts, I, bbox, cls, VOCopts.trainset, 0, 1);
%else
%end

