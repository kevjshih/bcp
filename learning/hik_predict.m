function pred = hik_predict(model, X)


pred = zeros(1,size(X,2)) + model.bias;

%inds = [1:size(X,2)]';

for i = 1:size(X,1)
   quantX = max(0,min(floor(X(i,:)*model.scaling(i)), model.Nbins-1)); % 0 indexed
   pred = pred + model.Ml(i, quantX+1).*X(i,:);
   pred = pred + model.Mu(i, quantX+1);
end
