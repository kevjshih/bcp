function f = fobj_bbox_weighting(w, targets, data)

w = reshape(w, [], 4); 

f0 = zeros(length(targets), 4);

for i = 1:length(targets)
   B = data{i}(:, 1:4);
   ps = ones(size(data{i}(:, end)));
   
   num = diag(w'*bsxfun(@times, data{i}, ps));
   Z = w'*ps;
   
   pred = num./Z;

   f0(i, :) = (pred' - targets{i}).^2;
end

%f = f0;%
f = sum(f0(:));
