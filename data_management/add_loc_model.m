function model = add_loc_model(model, init)

if(~exist('init','var'))
   model.part(end+1).filter = zeros(8, 8, 31);
else
   model.part(end+1).filter = init;
end
model.part(end).size = size(model.part(end).filter);
model.part(end).bias = 0;
model.part(end).computed = 0;
model.num_parts = model.num_parts + 1;
