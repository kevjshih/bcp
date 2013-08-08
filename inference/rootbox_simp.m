function boxes = rootbox(x, y, scale, padx, pady, rsize)
x1 = (x(:)-padx).*scale(:)+1;
y1 = (y(:)-pady).*scale(:)+1;
x2 = x1 + rsize(2).*scale(:) - 1;
y2 = y1 + rsize(1).*scale(:) - 1;

boxes = [x1 y1 x2 y2];

