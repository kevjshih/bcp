function p = add_whitening(p)

% Construct whitening info
S = construct_cov(p.size);
S = 1/2*(S + S'); % Make symmetric

p.whiten.W = S^(-1/2);
p.whiten.mu = construct_bg_mean(p.size);

% Initialize model so that after whitening results are the same:
filter0 = p.filter;
p.filter = reshape(S^(1/2)*p.filter(:), size(p.filter));
p.bias = p.bias + filter0(:)'*p.whiten.mu(:);
