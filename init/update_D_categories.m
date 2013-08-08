function D = update_D_categories(D, cls)

switch cls
   case {'object'}
      for i = 1:length(D)
         if(isfield(D(i).annotation, 'object') && ~isempty(D(i).annotation.object))
            [D(i).annotation.object.name] = deal(cls);
         end
      end
end
