function[labels cached] = ...
    prune_boost_data_overlap_fixed_negatives(D, cached_scores, cls, neg_inds_all)

for i = 1:length(cached_scores)
    all_inds = [];
    if(isempty(cached_scores{i}.labels))
        continue;
    end
    annotation = D(i).annotation;
    
    big_enough = ones(size(cached_scores{i}.scores)); %mean(isinf(cached_scores{i}.part_scores),2)<=1;%0.85;
    
    ok_pos = find(cached_scores{i}.labels>0 & big_enough);
    if(any(cached_scores{i}.labels>0)) % positive image
        boxes = LMobjectboundingbox(annotation, cls);
        [overlaps best_ind] = max(bbox_overlap_mex(boxes, cached_scores{i}.regions(ok_pos, :)), [], 2);
        
        pos_inds = ok_pos(best_ind(overlaps>0.5));
        all_inds = pos_inds;
    end 

    all_inds = [all_inds; neg_inds_all{i}(:)];
    %cached{i} = [cached_scores{i}.part_scores(all_inds,:) all_inds(:)];
    if(isfield(cached_scores{i}, 'part_scores'))
        cached{i} = [cached_scores{i}.part_scores(all_inds,:) cached_scores{i}.region_score(all_inds,:) all_inds(:)];
    else
        cached{i} = [cached_scores{i}.region_score(all_inds,:) all_inds(:)];
    end
    
    labels{i} = cached_scores{i}.labels(all_inds);
    %scores{i} = cached_scores{i}.scores(all_inds);
    % regions{i} = cached_scores{i}.regions(all_inds,:);
    %imind{i} = repmat(i, length(all_inds), 1);
end

labels = cat(1, labels{:});
labels(labels>0) = 1;
labels(labels<0) = -1;
cached = cat(1, cached{:});
%scores = cat(1, scores{:});
%imind = cat(1, imind{:});

function best_ind = get_best_hyp(true_label, scores)

[ind dk un_label] =  unique(true_label);

for i = 1:length(ind)
    this_ind = find(un_label==i);
    [best_score best_ind_t] = max(scores(this_ind));
    best_ind(i) = this_ind(best_ind_t);
end


    