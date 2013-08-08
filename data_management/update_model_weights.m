function model = update_model_weights(model, w)

if(isfield(model, 'loc_model') && model.loc_model==1)
   todo = find(~[model.part.computed]);
   model.part(todo).filter = reshape(w(1:prod(model.part(todo).size)), model.part(todo).size);
   model.part(todo).bias = w(end);
   model.cached_weight = reshape(w(prod(model.part(todo).size) + (1:numel(model.cached_weight))), 1, []);
else
bias = w(end);
num_computed = sum([model.part.computed]==0);
per_part_bias = bias/num_computed; % This is only valid because every part is always detected.  This may need to be fixed later

baseind = 0;
for i = 1:model.num_parts
    part = model.part(i);
    if(part.computed==0) % Take off the next chunk
        % Appearance model
        
        if(iscell(part.filter)) % Spatial model isn't implemented here!
            for sub = 1:length(part.filter)
                chunk_size = prod(size(part.filter{sub}));
                model.part(i).filter{sub} = reshape(w(baseind + [1:chunk_size]), size(part.filter{i}));
                baseind = baseind + chunk_size;
            end
            chunk_size = numel(part(i).bias);
            model.part(i).bias = per_part_bias + reshape(w(baseind + [1:chunk_size]), 1, chunk_size);
            baseind = baseind + chunk_size;
        else
            chunk_size = prod(size(part.filter));
            model.part(i).filter = reshape(w(baseind + [1:chunk_size]), size(part.filter));
            baseind = baseind + chunk_size;
        
            if(isfield(part, 'spat_w') && ~isempty(part.spat_w))
                chunk_size = prod(size(part.spat_w));
                model.part(i).spat_w = reshape(w(baseind + [1:chunk_size]), size(part.spat_w));
                baseind = baseind + chunk_size;
    
                chunk_size = prod(size(part.scal_w));
                model.part(i).scal_w = reshape(w(baseind + [1:chunk_size]), size(part.scal_w));
                baseind = baseind + chunk_size;
            end
        
            model.part(i).bias = per_part_bias;
        end
    end
end
            
model.cached_weight = reshape(w(baseind + [1:length(model.cached_weight)]), 1, []);
baseind = baseind + length(model.cached_weight);
if((baseind+1)~=(length(w)))
    error('Something doesn''t line up when updating model weights!\n');
end

end
