function correct = manual_get_applicability(applicability, ims, obj_box)
% Returns whether each object is applicable (correct ~= -2 for each
% object that is applicable).
%
% Use an applicability created from 'manual_set_applicability'.

correct = zeros(length(ims), 1);
parfor ims_i = 1:length(ims)
   correct(ims_i) = single_applicability_lookup(applicability.lookup, ims{ims_i}, obj_box{ims_i});
end
end

function correct = single_applicability_lookup(lookup, im, obj_box)
% Returns whether the single object is applicable to the part-lookup.

[dc target_name] = fileparts(im);
% Only use y-coordinates because x may be flipped.
target_obj_box = obj_box([2 4]);
target_obj_box_str = mat2str(target_obj_box);
target_obj_box_str = strrep(target_obj_box_str, ' ', '_');
target_obj_box_str = target_obj_box_str(2:end-1);  % Strip '[' and ']'

field = ['x' target_name target_obj_box_str];  % Add 'x' so it's a valid field name.
if isfield(lookup, field)
   correct = lookup.(field);
else
   disp(['Warning: Could not find applicability for ' target_name ' ' mat2str(target_obj_box)]);
   correct = 0;
end
end