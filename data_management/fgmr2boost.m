function model = fgmr2boost(model, fgmr)

if(isfield(fgmr, 'class')) % This is a V4 model
   % Collect object level (seem to always appear on LHS of rules)
   roots = fgmr.rules{fgmr.start}; 

   for i = 1:length(roots)
      rhs = roots(i).rhs(1);

      if fgmr.symbols(rhs).type == 'T'
         % handle case where there's no deformation model for the root
         root_ind(i) = fgmr.symbols(rhs).filter;
      else
         % handle case where there is a deformation model for the root
         root_ind(i) = fgmr.symbols(fgmr.rules{rhs}(1).rhs).filter;
      end
   end

   for i = 1:length(fgmr.filters)
      model.part(end+1).filter = fgmr.filters(i).w(:,:,1:31);
      model.part(end).size = fgmr.filters(i).size;
      model.part(end).bias = 0;
      model.part(end).computed = 0;

      if(ismember(i, root_ind))
         model.part(end).spat_const = [0 1 0 1 0.50 1];
      else
         model.part(end).spat_const = [0 1 0.8 1 0.1 1];
      end

      model.num_parts = model.num_parts + 1;
   end
else
   for i = 1:length(fgmr.rootfilters)
      model.part(end+1).filter = fgmr.rootfilters{i}.w;
      model.part(end).size = fgmr.rootfilters{i}.size;
      model.part(end).bias = 0;
      model.part(end).computed = 0;
      model.part(end).spat_const = [0 1 0 1 0.50 1];
      model.num_parts = model.num_parts + 1;
   end

   for i = 1:length(fgmr.partfilters)
      if(fgmr.partfilters{i}.fake)
         continue;
      end
   
      model.part(end+1).filter = fgmr.partfilters{i}.w;
      model.part(end).size = size(fgmr.partfilters{i}.w);
      model.part(end).bias = 0;
      model.part(end).computed = 0;
      model.part(end).spat_const = [0 1 0.8 1 0.1 1];
      model.num_parts = model.num_parts + 1;
   end
end


model.maxsize = max(cat(1, model.part.size),[], 1);
