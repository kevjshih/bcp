function cont = contained_center(A, B)
% Check to see if B is contains the center of A

center = 1/2*(A(:, [1 2]) + A(:, [3 4]));


cont_x = bsxfun(@ge, center(:, 1), B(:, 1)') & bsxfun(@le, center(:, 1), B(:, 3)');
cont_y = bsxfun(@ge, center(:, 2), B(:, 2)') & bsxfun(@le, center(:, 2), B(:, 4)');

cont = cont_x & cont_y;
