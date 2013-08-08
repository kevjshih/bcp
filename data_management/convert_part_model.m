function new_model = convert_part_model(model)

if(iscell(model))
   for i = 1:length(model)
      new_model{i} = convert_part_model(model{i});
   end

   return;
elseif(numel(model)>1)
   for i = 1:length(model)
      new_model(i) = convert_part_model(model(i));
   end

   return;
end

if(isfield(model, 'model')) % exemplar svm has nestest structure
    new_model = convert_part_model(model.model);
elseif(isfield(model, 'filter')) % mine to exemplar
   new_model.w = model.filter;
   new_model.hg_size = model.size;
   new_model.b = model.bias;
elseif(isfield(model, 'w')) % Convert from exemplar to mine
   new_model.filter = model.w;
   new_model.size = model.hg_size;
   new_model.bias = -model.b;
   
   if(isfield(model, 'bb'))
       new_model.bb = model.bb;
   end
end

if(isfield(model, 'name'))
   new_model.name = model.name;
elseif(isfield(model, 'models_name')) % Exemplar svm fieldname
   new_model.name = model.models_name;
end
