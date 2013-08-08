function pred = hik_predict_exact(Xte, Xtr, alphas, labels)

svs = find(alphas>1e-9);

pred = zeros(1, size(Xte,2));

for sv = svs(:)'
   pred = pred + alphas(sv)*labels(sv)*sum(bsxfun(@min, Xte, Xtr(:, sv)));
end

