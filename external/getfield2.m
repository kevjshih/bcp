function res = getfield2(str, varargin)
% Recursively get fields from complex structures
% Usage: getfield2(structure, field1, field2, ...)
% To access each element (if fieldx is an array), set the following argument as []
% 
% output = getfield2(D, [], 'annotation','object', [], 'polygon')
%  Will access every element from D(i).annotation.object(j).polygon
%  And append them in a nested cell array
%  i.e. output{i}{j} = D(i).annotation.object(j).polygon
%
%  these can then be concatenated together if the types are compatible
%  i.e. output_t = cat(dim, output{:});
%       output = cat(dim, output_t{:});

if(ischar(varargin{1}))
   res = str.(varargin{1});
   
   if(nargin>2)
      res = getfield2(res, varargin{2:end});
   end
elseif(isempty(varargin{1}))
   if(nargin>2)
      for i = 1:length(str)
         res{i} = getfield2(str(i), varargin{2:end});
      end

      % If the results are compatible, concatenate them as an array
      try
         if(length(res)>0 && ~ischar(res{1}))
            res = [res{:}];
         end
      end
   else
      for i = 1:length(str)
         res(i) = str(i);
      end
   end

elseif(isdouble(varargin{1}))
   els = varargin{1};
   res = {};
   for i = 1:length(els)
      res{i} = str(els(i));
   end
      
   % If the results are compatible, concatenate them as an array
   try
      res = [res{:}];
   end
end

