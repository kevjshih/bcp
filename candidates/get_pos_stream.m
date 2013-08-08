function ps = get_pos_stream(VOCopts, set, class)
% ps = get_pos_stream(VOCopts, set, class)
%
% Get positive examples for a class from a PASCAL set.
% Modified from get_pascal_stream.m
%
% Input:
%   VOCopts: parameters for the dataset
%   set: PASCAL set to retrieve from
%   class: class of object for which to get positive examples
%
% Output:
%   ps: cell array of structs with the fields...
%         I: string of image path
%         bbox(N, [x1 y1 x2 y2]): ground-truth boxes, one per row
%         cls: string of the object's class
%         id: the PASCAL id of the image
%

basedir = sprintf('%s/pos streams', VOCopts.localdir);
if ~exist(basedir,'dir')
    mkdir(basedir);
end

cached_filename = sprintf('%s/%s-%s.mat', basedir, set, class);
if fileexists(cached_filename)
    load(cached_filename, 'ps');
    return;
end

ps = {};
[ids,gt] = textread(sprintf(VOCopts.clsimgsetpath,class,set),'%s %d');
ids = ids(gt==1);

for i = 1:length(ids)
    curid = ids{i};
    
    recs = PASreadrecord(sprintf(VOCopts.annopath,curid));    
    filename = sprintf(VOCopts.imgpath,curid);
    
    bbox = [];
    for objectid = 1:length(recs.objects)
        if ~ismember({recs.objects(objectid).class},{class})
            continue;
        end
        bbox = [bbox; recs.objects(objectid).bbox];
    end

    res.I = filename;
    res.bbox = bbox;
    res.cls = class;
    res.id = curid;
    ps{end+1} = res;
end

save(cached_filename, 'ps');
end
