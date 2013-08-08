function chosen = choose_candidate_random(candidate_models, part_proportion)

if(~exist('part_proportion', 'var'))
   part_proportion = 0.75;
end

% First, figure out which are 
for i = 1:length(candidate_models)
   ispart(i) = ~((isfield(candidate_models{i}, 'spat_const') && ~isempty(candidate_models{i}.spat_const) && candidate_models{i}.spat_const(5)>0.5));
end

part_inds = find(ispart);
obj_inds = find(~ispart);

part_counter = 1;
obj_counter = 1;
for i = 1:100
   if(mod(i, ceil(1/(1-part_proportion)))==1)
      chosen(i) = obj_inds(obj_counter);
      obj_counter = obj_counter + 1;
   else
      chosen(i) = part_inds(part_counter);
      part_counter = part_counter + 1;
   end
end
