function S = construct_cov(sz, cov_cell)

if(~exist('cov_cell', 'var'))
   [func_base] = fileparts(which('construct_cov.m'));
   load(fullfile(func_base, 'data/cov_cell.mat'), 'cov_cell');
end

Ny = (size(cov_cell,1)+1)/2;
Nx = (size(cov_cell,2)+1)/2;
Nf = size(cov_cell,3);

N.Ny = sz(1);
N.Nx = sz(2);
N.Nf = Nf;

% Indexing isn't going to be pretty
S = zeros(sz(1)*sz(2)*Nf, sz(1)*sz(2)*Nf);
visited = S;

% Starting with the slow but safe method
for y1 = 1:sz(1)
   for y2 = max(1,y1-Ny+1):min(sz(1), y1+Ny-1)
      for x1 = 1:sz(2)
         for x2 = max(1,x1-Nx+1):min(sz(2), x1+Nx-1)
            S(Is(y1,x1,N), Is(y2,x2,N)) = reshape(cov_cell(y1-y2+Ny, x1-x2+Nx, :, :), [Nf, Nf]);
            visited(Is(y1,x1,N), Is(y2,x2,N)) = visited(Is(y1,x1,N), Is(y2,x2,N)) + 1;
         end
      end
   end
end

S = S + 0.01*eye(size(S)); % Apply smoothing


function inds = Is(y, x, N)
% This should line up with sub2ind (which it seems to)

inds = y+(x-1)*N.Ny+([1:N.Nf]-1)*N.Ny*N.Nx;




