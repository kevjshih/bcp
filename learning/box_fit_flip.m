function pred = box_fit(theta, x)

pred_dims = x(:, 1:end/3); 
probs = x(:, end/3+1:2*end/3);
flipped = x(:, 2*end/3+1:end);

pred = ((pred_dims.*probs)*theta)./(probs*theta);
pred = pred(:);

%pred = ((pred_dims.*probs)*theta)./sum(probs,2);
