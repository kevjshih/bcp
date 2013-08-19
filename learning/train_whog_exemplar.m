function m = train_whog_exemplar(m)

S = construct_cov(m.hg_size);
Sinv = inv(S);

mu0 = construct_bg_mean(m.hg_size);

m.w = reshape(Sinv*(m.x(:) - mu0(:)), [m.hg_size(1:2) 31]);
