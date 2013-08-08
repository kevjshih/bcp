function train_candidate_parts(cls, directory, substr)

VOCinit;
VOCopts.sbin = 8;

%ps = get_pos_stream(VOCopts, set, cls);
%[I bbox] = auto_get_part(VOCopts, ps, num);
[I bbox] = get_part_info(directory, substr, VOCopts);
models = orig_train_exemplar(VOCopts, I, bbox, cls, VOCopts.trainset, 1);




function [I bbox] = get_part_info(directory, substr, VOCopts)

d = dir(fullfile(directory, [substr '.mat']));

for i = 1:length(d)
%2008_005247-[271 150 311 184]-train-10000.mat
   str = d(i).name;
   %t = load(fullfile(directory, str),'m');

   [I0 bbox{i}] = extract_exemplar_params(str);
   
   [dk I0 ext] =  fileparts(I0);
   I{i} = sprintf(VOCopts.imgpath, I0);
   %bbox{i} = t.m.bbox;
end
