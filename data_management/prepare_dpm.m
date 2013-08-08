function model = prepare_dpm(model)

model.cls = model.class;
model.do_transform = 1;



root_filters = model.rules{model.start};

for i = 1:length(root_filters) % This is a very shallow parse, but I think it should be general enough for anything we do
   for j = 1:length(root_filters(i).rhs)
      filt_symb_ind = model.rules{root_filters(i).rhs(j)}.rhs;
      filt_ind = model.symbols(filt_symb_ind).filter;
      model.neighbors(i,j) = filt_ind;
   end
end

model.num_parts = numel(model.neighbors)/2;%length(model.rules{model.start})/2; % L/R flip
[model.part(1:model.num_parts).computed] = deal(0);
model.do_boxes = 1;
