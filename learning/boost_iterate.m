function [new_learner all_learners roc_50_t] = boost_iterate(D, cached_scores, cls, num_iter, varargin)
%function [new_learner all_learners] = boost_iterate(D, cached_scores, cls, num_iter, varargin)

   best_iter = inf;
   best_score = -inf;

   for boost_iter = 1:num_iter
      if(boost_iter == 1) % Start from scratch every time.
         [labels_sub cached_sub imind] = prune_boost_data_overlap(D, cached_scores, cls);
      else % Search for highest scoring region
        [labels_sub0 cached_sub0 imind0] = prune_boost_data(D, cached_scores, cls);

         update_pos = 1;%%mod(boost_iter-1,5)==0
        %[labels_sub cached_sub imind] = update_boost_set(labels_sub, cached_sub, imind, labels_sub, cached_sub, imind, 0, new_learner);
        [labels_sub cached_sub imind] = update_boost_set(labels_sub, cached_sub, imind, labels_sub0, cached_sub0, imind0, update_pos, new_learner);
      end

%      if(boost_iter==1)
%         % Remove rank feature
%         if(numel(varargin)>=2)

      fprintf('Cache size: %d\n', numel(labels_sub));
      new_learner = boost_train(cached_sub, labels_sub, varargin{:});

%      if(nargout>=2)
         all_learners{boost_iter} = new_learner;
%      end

      if(num_iter>1)
         cached_scores = apply_weak_learner(cached_scores, new_learner);
         roc_50_t(boost_iter) = test_given_cache(D, cached_scores, cls, [0.5]);

         if(roc_50_t(boost_iter).ap >= best_score)
            best_score = roc_50_t(boost_iter).ap;
            best_iter = boost_iter;
         end

         if(best_iter+2 <= boost_iter)
            fprintf('Iter:%d Model hasn''t improved since iteration %d, terminating\n', boost_iter, best_iter);
            break;
         end
      end
   end

   if(num_iter==1)
      new_learner = all_learners{1};
   else
      [best_ap new_learner_ind] = max([roc_50_t.ap]); % Choose learner with the max ap
      new_learner = all_learners{new_learner_ind};
      fprintf('Chose iteration %d, with AP:%f\n', new_learner_ind, best_ap);
   end
