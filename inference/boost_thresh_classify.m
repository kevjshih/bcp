function [ output ] = classify_with_boosting( data, boost_struct)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

output = zeros(size(data,1),1);

if(numel(data)==0)
    return;
end

thresh_chosen = boost_struct.thresholds_chosen;

F = boost_struct.F;
for m = 1:size(thresh_chosen,1),
    
    feat = thresh_chosen(m,1);
    thresh = (thresh_chosen(m,2));
    data_tmp = zeros(size(data(:,feat)));
    data_tmp(data(:,feat) <= thresh) = 1;
    data_tmp(data(:,feat) > thresh) = 2;

    output = output + F(m, data_tmp)';
%    for i = 1:size(data,1),
%        output(i) = output(i) + F(m,data_tmp(i));
%    end
end

end

