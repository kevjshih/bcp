function [model, cached_scores] = combine_part_models(part_models, part_cached_scores)
% Combine object models and cached scores that each only reference a single part model.
%
% "part_models" and "part_cached_scores" should be cell arrays, with each index corresponding to the same part.
% They must be non-empty.

%% Combine models
model = part_models{1};
for i = 2:length(part_models)
   curr_part_model = part_models{i};
   model.num_parts = model.num_parts + curr_part_model.num_parts;
   for j = 1:curr_part_model.num_parts
      model.part(end+1) = curr_part_model.part(j);
   end
end

if ~exist('part_cached_scores', 'var')
   return;
end

%% Combine cached_scores
cached_scores = part_cached_scores{1};

% Accumulate all the parts-related field names in cached_scores.
cached_scores_fields = fieldnames(cached_scores{1});
part_fields = {};
for i = 1:length(cached_scores_fields)
   field = cached_scores_fields{i};
   if strfind(field, 'part_') == 1
      % The field started with 'part_'.
      part_fields{end+1} = field;
   end
end

for i = 2:length(part_cached_scores)
   % Append for cached_scores
   curr_part_cached_scores = part_cached_scores{i};
   for j = 1:length(cached_scores)
      for k = 1:length(part_fields)
         part_field = part_fields{k};
         cached_scores{j}.(part_field) = [cached_scores{j}.(part_field) curr_part_cached_scores{j}.(part_field)];
      end
   end
end

end
