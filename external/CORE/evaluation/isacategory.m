function res = isacategory(tocheck, query)

cats = getCategoryInfo(tocheck);

if(~iscell(query))
    query = {query};
end

for i = 1:length(cats)
   res(i) = any(ismember(cats{i}, query));
end