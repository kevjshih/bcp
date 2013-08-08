function [new_learner all_learners roc_50_t] = boost_iterate(D, cached_scores, cls, num_iter, varargin)
%function [new_learner all_learners] = boost_iterate(D, cached_scores, cls, num_iter, varargin)
   if(length(varargin)>=1)
      columns = varargin{1};
   else
      columns = [];
   end


   for boost_iter = 1:num_iter
      if(boost_iter == 1) % Start from scratch every time.
         [labels_sub cached_sub imind] = prune_boost_data_overlap(D, cached_scores, cls);

%         for i = 1:length(cached_scores)
%            cached_scores{i}.score = [1:length(cached_scores{i}.scores)];
%         end
%        [labels_sub cached_sub imind] = prune_boost_data(D, cached_scores, cls);
      else % Search for highest scoring region
        [labels_sub cached_sub imind] = prune_boost_data(D, cached_scores, cls);

        % update_pos = 1;%%mod(boost_iter-1,5)==0
        %[labels_sub cached_sub imind] = update_boost_set(labels_sub, cached_sub, imind, labels_sub, cached_sub, imind, 0, new_learner);
        %[labels_sub cached_sub imind] = update_boost_set(labels_sub, cached_sub, imind, labels_sub0, cached_sub0, imind0, update_pos, new_learner);
      end

%      if(boost_iter==1)
%         % Remove rank feature
%         if(numel(varargin)>=1)
%           varargin{1}(varargin{1}==size(cached_sub,2)) = [];
%         else
%           varargin{1} = 1:size(cached_sub,2)-1;
%         end
%      else
%         varargin{1} = columns;
%      end

      fprintf('Cache size: %d\n', numel(labels_sub));
      new_learner = boost_train(cached_sub, labels_sub, varargin{:});

      if(nargout>=2)
         all_learners{boost_iter} = new_learner;
      end

      if(num_iter>1)
         cached_scores = apply_weak_learner(cached_scores, new_learner);
         roc_50_t(boost_iter) = test_given_cache(D, cached_scores, cls, [0.5]);
      end
   end

   if(num_iter==1)
      new_learner = all_learners{1};
   else
      [best_ap new_learner_ind] = max([roc_50_t.ap]); % Choose learner with the max ap
      new_learner = all_learners{new_learner_ind};
      fprintf('Chose iteration %d, with AP:%f\n', new_learner_ind, best_ap);
   end
