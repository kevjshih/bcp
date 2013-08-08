function pred = box_fit(theta, x)

pred_dims = x(:, 1:end/2); 
probs = x(:, end/2+1:end);

pred = ((pred_dims.*probs)*theta)./(probs*theta);


%pred = ((pred_dims.*probs)*theta)./sum(probs,2);
