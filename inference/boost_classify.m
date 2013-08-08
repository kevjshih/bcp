function output = boost_scores(feat, learner)

learner_type = learner.type;

%feat(isinf(feat)) = -10000;


switch learner_type
    case 'svm'
        if(isempty(feat))
            output = [];
        else
            toosmall = mean(isinf(feat),2)>0.8;
            feat(:,end) = feat(:, end)/500;
            feat = feat(:, learner.columns);

            feat_approach = 'b';
            switch feat_approach
               case 'a'
                  feat = [feat isinf(feat)]';
               case 'b'
                  feat = [feat mean(isinf(feat),2)]';
               case 'c'
                  feat = [feat]';
            end

            feat(isinf(feat)) = 0;
            output = reshape(learner.model'*feat, [], 1);
%            output(toosmall) = -10000;
        end
   case 'thresh'
      output = boost_thresh_classify(feat, learner);
   case 'binned'
      output = logit_classify(feat, learner);
   case 'dt'
      if(isempty(feat))
         output = [];
      else
%         output = test_boosted_dt_mc(learner.model, feat(:, [1:end-2 end]));
%         feat = feat(:,[1 2 3 4 6]);
         %feat = [feat isinf(feat)];
         output = test_boosted_dt_mc(learner.model, feat(:,learner.columns));
      end
    case {'thresh_java', 'sigmoid_java', 'sigmoid_java_inf', ...
          'sigmoid_layered_java', 'sigmoid_java_co', 'sigmoid_inf_java'}
      if(isempty(feat))
         output = [];
      else
%         output = test_boosted_dt_mc(learner.model, feat(:, [1:end-2 end]));
%         feat = feat(:,[1 3]);
         feat(:,end) = -feat(:, end);
         [data] = process_data_java(feat);
         data = single(data);
      
         output = learner.model.classify(data);

%         output(mean(isinf(data),2)>0.8) = -10000;
      end
end


