function model = add_model(model, toadd, spat)

if(isempty(model))
   model.interval = 10;
   model.sbin = 8;
   model.fixed = false;
   model.maxsize = [10 10];
   model.num_parts = 0;
   model.thresh = -1;
   model.part = [];
end


model.part(end+1).filter = toadd.w;
model.part(end).size = size(toadd.w);
model.part(end).bias = -toadd.b;
model.part(end).computed = 0;
model.num_parts = model.num_parts + 1;
model.spat_const = [];

if(exist('spat', 'var') && spat==1)
   model.part(end).spat_w = zeros(5,5); % Should probably be an odd number
   model.part(end).scal_w = zeros(3,1);

   if(~isfield(model, 'spat_weight'))
      model.spat_weight = 10;
   end
end


ok_to_copy = {'spat_const', 'name'};

for cp_field = ok_to_copy
   if(isfield(toadd, cp_field{1}))
       model.part(end).(cp_field{1}) = toadd.(cp_field{1});
   end
end
