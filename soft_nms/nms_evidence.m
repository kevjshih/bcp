function cached_scores = nms_evidence(cached_scores, boost_learner)

new_scores = cell(length(cached_scores), 1);

parfor i = 1:length(cached_scores)
   fprintf('%d\n', i);
   if(~isempty(cached_scores{i}.regions))
      new_scores{i} = nms_evidence_helper(cached_scores{i}, boost_learner);
   end
end

for i = 1:length(cached_scores)
   cached_scores{i}.nms_scores = new_scores{i};
end


function final_scores = nms_evidence_helper(cached_scores, boost_learner)

Nreg = size(cached_scores.regions, 1);
Nparts = size(cached_scores.part_scores,2);
feats = [cached_scores.part_scores cached_scores.region_score (1:Nreg)'];

% Get contributions to score from each column
columns = double(sort(boost_learner.model.getColumnsUsed+1));

contribution = zeros(size(feats));

%init_score = boost_classify(feats, boost_learner);

% Find the contribution of each part:
for i = columns(:)'
   featT = -inf*ones(Nreg+1, size(feats,2));
   featT(:, i) = [-inf; feats(:, i)];

   contributionT = boost_classify(featT, boost_learner);
   contribution(:, i) = reshape(contributionT(2:end), [], 1) - contributionT(1);
end

% Compute coocc matrix for each part
overlaps = zeros(Nreg, Nreg, size(cached_scores.part_scores,2));

for i = columns(columns<=size(cached_scores.part_scores,2))'
   if(~isempty(cached_scores.part_boxes))
      boxes = double(cached_scores.part_boxes(:, (1:4) + 4*(i-1)));
      overlaps(:, :, i) = bbox_overlap_mex(boxes, boxes)>0.9; 
   else % Part boxes aren't computed, so just find parts with the same score
      overlaps(:, :, i) = abs(bsxfun(@minus, cached_scores.part_scores(:, i), cached_scores.part_scores(:, i)'))<1e-9;
   end
end

ignored = zeros(size(feats));
chosen = zeros(Nreg,1);
final_scores = zeros(Nreg, 1);
ordering = zeros(Nreg,1);
% Do greedy search
for i = 1:Nreg
   cur_scores = sum(contribution.*(1-ignored),2);
   [dk chosenT] = max(cur_scores(~chosen));
   notchosen = find(~chosen);
   chosen_ind = notchosen(chosenT);
   chosen(chosen_ind) = 1;
   ordering(chosen_ind) = i;
   final_scores(chosen_ind) = cur_scores(chosen_ind);
   % Now suppress any shared parts
   suppress = shiftdim(overlaps(chosen_ind, ~chosen, :),1);
   ignored(~chosen, 1:Nparts) = ignored(~chosen, 1:Nparts) | (suppress & contribution(~chosen, 1:Nparts)>0);
end
