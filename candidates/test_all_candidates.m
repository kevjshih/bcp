function test_all_candidates(num_blocks1, num_blocks2, cls0, suffix, trainset)
   
   if(~exist('trainset', 'var'))
      trainset = 0;
   end


   cls = cls0; %'aeroplane'; % Doesn't matter which one we use
   if(trainset==0)
      load_init_data;
      suffix = [suffix '_train'];
   else
      suffix = [suffix '_trainval'];
      load_init_final;
   end



   [dk pos_inds] = LMquery(D, 'object.name', cls0, 'exact');
   neg_inds = 1:length(D);
   neg_inds(pos_inds) = [];
   neg_inds = neg_inds(1:min(end, max(200, length(pos_inds))));

   D = D([pos_inds(:); neg_inds(:)]);
   cached_scores = cached_scores([pos_inds(:); neg_inds(:)]);

   cached_gt = get_gt_pos_reg(D, cached_scores, cls0);

   if(~exist('suffix', 'var'))
      suffix = '';
   end

   % Break it into chunks (can't do all 3000 at once!)
   for i = 1:num_blocks1
      test_candidate_detections(D, cached_gt, cls0, [], num_blocks1, i, suffix, 'auto_exemplars');%num_blocks, block_id);
   end

   for i = 1:num_blocks2
      test_candidate_detections(D, cached_gt, cls0, [], num_blocks2, i, suffix, 'object_exemplars');%num_blocks, block_id);
   end
