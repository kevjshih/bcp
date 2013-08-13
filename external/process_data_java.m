function [ data, learners] = process_data_java( data, part_box, bbox, labels, columns, classifier, monotonic)
%PROCESS_DATA Summary of this function goes here
%   Detailed explanation goes here

if(~exist('columns', 'var') || isempty(columns))
    % Do all columns
    columns = 1:size(data,2);
end

if(~exist('classifier', 'var'))
   classifier = 'thresh';
end


learners = java.util.ArrayList();

num_examples = size(data,1);
num_data = size(data,2);

if(nargin<4)
   return;
end
thresh_pnts = computeThreshPoints(data(:,columns) ,labels);

for i = 1:length(columns)
   d = data(:,columns(i));
   d(isinf(d)) = [];
   weight(i) = max(1./(std(d)+eps), .1);
end

%weight(end-1) = 1/32*weight(end-1);

if(~exist('monotonic', 'var') || isempty(monotonic))
   MONOTONIC = false;
else
    MONOTONIC = monotonic;
end

%learners.add(javaboost.weaklearning.BiasLearner());
for i = 1:numel(thresh_pnts),
    threshes = thresh_pnts{i};
    if isempty(threshes)
        continue;
    end

   switch classifier
      case 'thresh'
         learners.add(javaboost.weaklearning.SingleFeatureMultiThresholdedLearner(columns(i)-1, threshes, MONOTONIC(min(i,end))));
      case {'sigmoid', 'sigmoid_inf', 'sigmoid_co'}
        %learners.add(javaboost.weaklearning ...
        %              .SingleFeatureMultiThresholdedToSigmoidLearner(columns(i)-1,
        %              threshes, MONOTONIC(min(i,end)), weight(i),
        %              1));
learners.add(javaboost.weaklearning ...
                      .SingleFeatureMultiThresholdedToSigmoidLearner(columns(i)-1, threshes, MONOTONIC(min(i,end)), weight(i),0));
   end

   if(strcmp(classifier, 'sigmoid_inf'))
      conditions = java.util.ArrayList();
      conditions.add(javaboost.conditioning.IsNegInfinityConditional(columns(i)-1));
      learners.add(javaboost.weaklearning.ConditionalLearner(conditions, conditions.get(0)));
   end
end


end


function [thresh_pnts_all] = computeThreshPoints(data, labels)
thresh_pnts = [];
%crit_incr = 0.01;

num_std = 3;
for i = 1:size(data,2),
    disp(['analyzing threshold points for column: ' num2str(i)]);
    data_col = data(:,i);
    pos = data_col(labels == 1);
    neg = data_col(labels == -1);
    pos = pos(pos > -Inf);
    neg = neg(neg > -Inf);
    %p_mean = mean(pos);
    %p_std = std(pos);
    %n_mean = mean(neg);
    %n_std = std(neg);
    
    smallest = min(data_col);
    largest = max(data_col);
    
    
    
    lb = min(pos);
    ub = max(neg);
    if lb > ub,
        tmp = lb;
        lb = ub;
        ub = tmp;
    end
%     if p_mean > n_mean, % this really should be the case everytime
%         lb = min(p_mean-num_std*p_std, n_mean + num_std*n_std);
%         ub = max(p_mean-num_std*p_std, n_mean + num_std*n_std);
%     elseif p_mean == n_mean % nasty case
%         lb = min(data_col);
%         ub = max(data_col);
%     else
%         lb = min(p_mean+num_std*p_std, n_mean - num_std*n_std);
%         ub = max(p_mean+num_std*p_std, n_mean - num_std*n_std); 
%     end
%    new_pnts = [lb:crit_incr:ub];
%new_pnts = linspace(lb,ub, 100);
new_pnts = linspace(lb,ub, 100);
    thresh_pnts_all{i} = new_pnts;
end


%disp(thresh_pnts');
end

function [centers] = computePartCenters(partBoxes)
num_cols = size(partBoxes,2)/4;
centers = zeros(size(partBoxes,1),num_cols*2);

for i = 1:num_cols,
    centers(:,2*i-1) = (partBoxes(:,i+2) - partBoxes(:,i))/2;
    centers(:,2*i) = (partBoxes(:,i+3) - partBoxes(:,i+1))/2;
end

end



function [xydisp] = computeNormalizedPartCentersDispositions(partBoxes, regionBoxes)
num_cols = size(partBoxes,2)/4;
xydisp = zeros(size(partBoxes,1),num_cols*2);

for i = 1:num_cols,
    xydisp(:,2*i-1) = double(partBoxes(:,4*i-1) + partBoxes(:,4*i-3))/2-regionBoxes(:, 1);
    xydisp(:,2*i) = double(partBoxes(:,4*i) + partBoxes(:,4*i-2))/2 - regionBoxes(:,2);
end


end

function [mins] = computeAnds(data),
data = single(data);
num_parts = size(data,2);
num_examples = size(data,1);
mins = zeros(num_examples, sum([1:num_parts-1]),'single');

count = 1;
for i= 1:num_parts,
    for j = i+1:num_parts,
        mins(:,count) = min(data(:,i), data(:,j));
        count = count+1;
    end
end
end

function [rel_ud] = computeRelativeSpatialFeatures(pos),
% pos in format [x1 y1 x2 y2 ....xn yn]
num_parts = size(pos,2)/2;
num_examples = size(pos,1);
rel_ud = zeros(num_examples, sum([1:num_parts-1]));
col_count = 1;
for i = 1:num_parts,
    for j = i+1:num_parts,
        rel_ud(:,col_count) = sign(pos(:,2*i)-pos(:,2*j));
        col_count = col_count+1;        
    end
end
end

function [tb_feat, lr_feat] = computeSpatialFeatures(pos, bbox)
% bbox in format [x1 y1 x2 y2]
% pos in format [x y]
bbox_horiz_mid = (bbox(:,3) + bbox(:,1))/2; % midpoint in horizontal direction
bbox_vert_mid = (bbox(:,4) + bbox(:,2))/2; % midpoint in vertical direction

tb_feat = pos(:,2) - bbox_vert_mid;
lr_feat = pos(:,1) - bbox_horiz_mid;
% normalize the feature
tb_feat = tb_feat./(bbox(:,4) - bbox(:,2));
lr_feat = lr_feat./(bbox(:,3) - bbox(:,1));


end
