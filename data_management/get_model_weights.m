function w = get_model_weights(model)


if(isfield(model, 'loc_model') && model.loc_model==1)
   todo = find(~[model.part.computed]);
   w = [model.part(todo).filter(:); model.cached_weight(:); model.part(todo).bias];
else
   for p = 1:model.num_parts
       part = model.part(p);
       if(part.computed==0)
           if(iscell(part.filter))
               w{p} = [];
               for  sub = 1:length(part.filter)
                   w{p} = [w{p}; part.filter{sub}(:)];
               end
               w{p} = [w{p}; part.bias(:)];
               b{p} = 0;
           elseif(isfield(part, 'spat_w') && part.spat_w==1)
               w{p} = [part.filter(:); part.spat_w(:); part.scal_w(:)];
               b{p} = part.bias; % going to sum up all the biases
           else
               w{p} = [part.filter(:)];
               b{p} = part.bias; % going to sum up all the biases
           end
       end
   end

   w{end+1} = model.cached_weight(:);
   
   w = [cat(1, w{:}); sum(cat(1,b{:}))];
end

    
