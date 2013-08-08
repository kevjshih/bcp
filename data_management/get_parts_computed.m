function computed = get_parts_computed(model, cached_scores)
% Thorough check

computed = zeros(model.num_parts, 1);

for i = 1:length(cached_scores)
    if(~isempty(cached_scores{i}.regions))
        computed_t = any(any(cached_scores{i}.part_scores~=0, 1), 3);
        len = min(length(computed_t), length(computed));
        computed = computed_t(1:len)' | computed(1:len);
    end
end
