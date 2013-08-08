function [boxes rot] = rootbox_trans(x, y, scale, shift, rot, padx, pady, rsize, imsize)

x1 = (x(:)-padx + shift(1)).*scale(:)+1;
y1 = (y(:)-pady + shift(2)).*scale(:)+1;
x2 = x1 + rsize(2).*scale(:) - 1;
y2 = y1 + rsize(1).*scale(:) - 1;

if(rot==0)
   boxes = [x1 y1 x2 y2];
elseif(abs(rot)>=90)
   error('This rotation not supported');
else
   boxUL0 = [x1 y1];
   boxBL0 = [x1 y2];
   boxUR0 = [x2 y1];
   boxBR0 = [x2 y2];
   
   theta = rot*pi/180;

   corner00 = [max(0, imsize(1)*sin(-theta)), max(0, imsize(2)*sin(theta))];

   rot_mat = [cos(theta) sin(theta); -sin(theta) cos(theta)];

   boxUL = bsxfun(@minus, boxUL0, corner00)*rot_mat;
   boxBL = bsxfun(@minus, boxBL0, corner00)*rot_mat;
   boxUR = bsxfun(@minus, boxUR0, corner00)*rot_mat;
   boxBR = bsxfun(@minus, boxBR0, corner00)*rot_mat;
   
   
   xs = [boxUL(:, 1), boxBL(:, 1), boxUR(:, 1), boxBR(:, 1)];
   ys = [boxUL(:, 2), boxBL(:, 2), boxUR(:, 2), boxBR(:, 2)];
   boxes = [min(xs, [], 2) min(ys, [], 2), max(xs,[],2), max(ys,[],2)];
end
