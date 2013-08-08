function mu = construct_bg_mean(sz)
%bg_cell = [];
persistent bg_cell

if(isempty(bg_cell))
    fprintf('Loading!\n');
   [func_base] = fileparts(which('construct_bg_mean.m'));
   load(fullfile(func_base, 'data/bg_cell.mat'), 'bg_cell');
end

   bg_cell = 1/2*(bg_cell + flipfeat(bg_cell)); % Forgot to do this during training...
   mu = repmat(bg_cell, [sz(1:2) 1]);
