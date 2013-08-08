function model = init_loc_model(cls)

   model.num_parts = 0;
   model.thresh = -1;
   model.part = [];

   model.cls = cls;
   model.cached_weight = 0;
   model.loc_model = 1;
