function cached = update_cached_scores(cached, w)

for i = 1:length(cached)
   cached{i}.scores = cached{i}.scores*w;
end
