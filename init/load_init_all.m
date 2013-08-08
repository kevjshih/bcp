function load_init_all
% Precompute everything

BDglobals;
BDpascal_init;

%VOCopts.classes = {'cat','person','dog','bottle','car'};

for i = 1:length(VOCopts.classes)
   init_wrapper(VOCopts.classes{i});
end

function init_wrapper(cls)

load_init_test;
