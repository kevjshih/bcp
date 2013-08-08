
   %   [cur_lab features iminds] = update_cache_set(cur_lab, features, iminds, labels_new, feat_new, iminds_new, 0); % Don't update positive set, because there aren't any positives in this new data!
   bowmodel{iter} = svmtrain_workingset(cur_lab(:), features, sprintf('-h 0 -t 5 -c %f -w1 %f -w-1 %f', C, 1, 1), param); 
