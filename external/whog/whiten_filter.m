function filter = whiten_filter(filter0, W)

filter = reshape(W*filter0(:), size(filter0));
