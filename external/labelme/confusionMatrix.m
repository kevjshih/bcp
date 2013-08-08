function C = confusionMatrix(z1, z2)
% Normalized with respect to z1
% Input
%   z1 = true class for each sample
%   z2 = assigned class to each sample
% Output
%   C = confusion matrix, such that:
%     C(i,j) = percentage of times that class i (z1=i) is assigned to class j (z2=j)
%     sum(C(i,:)) = 1

n = max(z1);
m = max(z2);
n = max(n,m);
m = n;

C = zeros([n m]);
for i = 1:n
    for j = 1:m
        C(i,j) = sum((z1==i).*(z2==j)) / sum(z1==i);
    end
end

