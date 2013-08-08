function[cached_scores, D] = prune_to_dpm_subset(D, cached_scores, ...
                                                cls, dpm_positives)

numpos_all = length(dpm_positives); 
D_fnames = cat(1,D(:).annotation);
D_fnames = cat(1,D_fnames(:).filename);

flipped = cat(1, dpm_positives(:).flip);
dpm_positives = dpm_positives(~flipped);

dpm_fnames = cat(1, dpm_positives(:).im);
dpm_fnames = dpm_fnames(:, end-14:end);

for i = 1:length(D) % for all images in dataset
                    
    % find corresponding dpm images
    dpm_matches = (sum(abs(bsxfun(@minus, dpm_fnames, ...
                           D(i).annotation.filename)),2) == 0);

    % collect sampled bboxs in DPM
    dpm_matched = dpm_positives(dpm_matches);
    dpm_matched_bboxes = zeros(numel(dpm_matched),4);
    for j = 1:size(dpm_matched_bboxes,1)
        dpm_matched_bboxes(j,:) = [dpm_matched(j).x1, dpm_matched(j).y1, ...
                            dpm_matched(j).x2, dpm_matched(j).y2 ]; 
    end
    
    % now figure out which of the ground truths weren't sampled
    % parse bboxes in D
    objects_D = D(i).annotation.object;

    not_sampled_bboxes = [];
    % extract all boxes relevant to class
    [rel_bboxes_D, rel_bboxes_idx] = LMobjectboundingbox(D(i).annotation, cls);
    if isempty(rel_bboxes_D) % if no relevant boxes, move on
        continue;
    end
    % remove nonsampled bboxes frem D

    toRemove = [];
    for j = 1:size(rel_bboxes_D,1);        % for all relevant boxes
        curr_bbox = rel_bboxes_D(j,:);
        gt_bbox_matched = sum(abs(bsxfun(@minus, dpm_matched_bboxes, ...
                              curr_bbox)),2) == 0;
        % check for matches in the dpm sampled boxes
        if isempty(find(gt_bbox_matched)) % bbox not matched
            not_sampled_bboxes =[not_sampled_bboxes; curr_bbox];
            toRemove(end+1) = rel_bboxes_idx(j);
            % keep track of this one for later
        end
      
    end

    % not_sampled_gt tells us which relevant gt bboxes weren't sampled in the
    % DPM
    
    % now go through cached_regions and start removing regions that
    % have more than 10% overlap with the non-sampled
    
    if isempty(not_sampled_bboxes) % if all boxes were sampled
        continue;
    end

    D(i).annotation.object(toRemove) = [];

    % for each not sampled relevant bbox

    %else, remove all regions with more than 10% overlap with
    %non-sampled positives
    ovs = bbox_overlap(not_sampled_bboxes, ...
                       cached_scores{i}.regions);
    if isempty(ovs)
        continue;
    end
    assert(size(ovs,2) == size(not_sampled_bboxes,1));
    drop_these = ovs  > 0.1;
    drop_these = max(drop_these,[], 2);

    assert(length(drop_these) == size(cached_scores{i}.regions,1));
    cached_scores{i}.regions = cached_scores{i}.regions(~ ...
                                                      drop_these,:);
    cached_scores{i}.labels = cached_scores{i}.labels(~drop_these,: ...
                                                      );
    cached_scores{i}.scores = cached_scores{i}.scores(~drop_these,: ...
                                                      );
    cached_scores{i}.part_scores = cached_scores{i}.part_scores(~ ...
                                                      drop_these,:);
    cached_scores{i}.part_boxes = cached_scores{i}.part_boxes(~ ...
                                                      drop_these,:);
    cached_scores{i}.region_score = cached_scores{i}.region_score(~ ...
                                                      drop_these,:);

end




function [boundingbox bboxidx] = LMobjectboundingbox(annotation, varargin)
% boundingbox = LMobjectboundingbox(annotation, name) returns all the bounding boxes that
% belong to object class 'name'. Is it an array Ninstances*4
%
% boundingbox = [xmin ymin xmax ymax]

bboxidx = [];
[x,y,jc] = LMobjectpolygon(annotation, varargin{:});

Nobjects = length(x);
if Nobjects == 0
    boundingbox = [];
else

    boundingbox = zeros(Nobjects,4);
    for n = 1:Nobjects
        [xn yn] = getLMpolygon(annotation.object(jc(n)).polygon);
        boundingbox(n,:) = [min(x{n}) min(y{n}) max(x{n}) ...
                            max(y{n})];
        bboxidx(end+1) = jc(n);
    end
end
