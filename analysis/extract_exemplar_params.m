function [im bb] = extract_exemplar_params(model)

if(isstruct(model))
   name = model.name;
else
   name = model;
end


fn = regexp(name, '\d{4}_\d{6}', 'match');
bb0 = regexp(name, '\[\d+ \d+ \d+ \d+\]', 'match');
bb = str2num(bb0{1}(2:end-1));

im = [fn{1} '.jpg'];
