function cluster_test_candidates(cls, num_blocks, block_id)

%try

   if(matlabpool('size')==0)
      matlabpool local;
   end
   load_init_data;
 

if(~exist('block_id', 'var')) 
   for i = 1:num_blocks
      test_candidate_detections(D, cached_scores, cls, [], num_blocks, i);%num_blocks, block_id);
   end
else
   test_candidate_detections(D, cached_scores, cls, num_blocks, block_id);
exit;
end
%end


