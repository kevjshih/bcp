function model = init_model(cls)
   model.interval = 10;
   model.sbin = 8;
   model.fixed = false;
   model.maxsize = [10 10];
   model.num_parts = 0;
   model.thresh = -1;
   model.part = [];

   model.cls = cls;
   model.cached_weight = 1;
