function part = train_candidate_parts(part, cls)

BDglobals;
BDVOCinit;
VOCopts.sbin = 8;

%ps = get_pos_stream(VOCopts, set, cls);
%[I bbox] = auto_get_part(VOCopts, ps, num);
str = part.name;

[I0 bbox] = extract_exemplar_params(str);

[dk I0 ext] =  fileparts(I0);
I = sprintf(VOCopts.imgpath, I0);

part_t = orig_train_exemplar(VOCopts, I, bbox, cls, VOCopts.trainset, 1);

part = part_t.model;
part.name = part_t.models_name;



function [I bbox] = get_part_info(directory, substr, VOCopts)

