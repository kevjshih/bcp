function model = add_subset_model(model)

if(isfield(model, 'subset_split') && model.subset_split>0)
   model.part(end).filter = repmat({model.part(end).filter}, 1, model.subset_split);
   model.part(end).bias = repmat(model.part(end).bias, 1, model.subset_split);
else
   warning('Not actually adding anything!  Make sure subset_split field is set!\n');
end
