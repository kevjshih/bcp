function test_all_candidates_subset(D, cached_scores, num_blocks1, num_blocks2, cls0, numpos, ...
                             trial)
   



   [dk pos_inds] = LMquery(D, 'object.name', cls0, 'exact');
   neg_inds = 1:length(D);
   neg_inds(pos_inds) = [];
   neg_inds = neg_inds(1:min(end, max(200, length(pos_inds))));

   D = D([pos_inds(:); neg_inds(:)]);
   cached_scores = cached_scores([pos_inds(:); neg_inds(:)]);

   cached_gt = get_gt_pos_reg(D, cached_scores, cls0);

   % Break it into chunks (can't do all 3000 at once!)
   for i = 1:num_blocks1
      test_candidate_detections_subset(D, cached_gt, cls0, [], ...
                                       num_blocks1, i, 'auto_exemplars', ...
                                       numpos, trial);%num_blocks, block_id);
   end

   for i = 1:num_blocks2
      test_candidate_detections_subset(D, cached_gt, cls0, [], ...
                                       num_blocks2, i, 'object_exemplars', ...
                                       numpos, trial);%num_blocks, block_id);
   end
