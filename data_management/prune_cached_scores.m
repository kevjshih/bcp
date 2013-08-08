function new_cached = prune_cached_scores(cached_score, reg_inds, part_inds)


fields = fieldnames(cached_score);
numreg = size(cached_score.regions,1);

new_cached = cached_score;

for i = 1:length(fields)
   if(size(cached_score.(fields{i}),1)==numreg) % This can be pruned
      new_cached.(fields{i}) = cached_score.(fields{i})(reg_inds, :);
   end
end
