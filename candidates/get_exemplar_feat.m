function m = get_exemplar_feat(model)

BDglobals;

VOCinit;
VOCopts.sbin = 8;

params = VOCopts;
params.sbin = 8;
params.interval = 10;
params.MAXDIM = 10;


for i = 1:model.num_parts
   [I bbox] = extract_exemplar_params(model.part(i).name);
   I = fullfile(im_dir, I);
   m{i} = initialize_goalsize_model(convert_to_I(I), bbox, params);
end
