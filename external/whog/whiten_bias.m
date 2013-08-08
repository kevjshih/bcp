function bias = whiten_bias(filter, W, mu)

bias = filter(:)'*W*mu(:);
