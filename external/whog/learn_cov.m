function learn_cov

% Use pascal train set
addpath(genpath('~/prog/VOC2010/VOCdevkit/VOCcode'));
addpath('~/prog/voc-release3.1');
VOCopts = [];
VOCinit;

ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');

sbin = 8; interval = 10; % Not subsampling pyramid

Nf = 31;
Nx = 16;
Ny = 16;

cov_cell = zeros(Ny*2-1, Nx*2-1, Nf, Nf); % This will allow a full covariance for a 16x16 model

parfor i = 1:length(ids)
   fprintf('%d/%d\n', i, length(ids));
   im = imread(sprintf(VOCopts.imgpath, ids{i}));

   pyr = featpyramid(im, sbin, interval);
   
   im_cov_cell = zeros(Ny*2-1, Nx*2-1, Nf, Nf); % This will allow a full covariance for a 16x16 model
   im_counts = zeros(Ny*2-1, Nx*2-1);

   % Doing this the slow but safe way
   for j = 1:length(pyr)
      for flip = 1:2
         F = pyr{j}; % The features at this level
         if(flip==2)
            F = flipfeat(F);
         end

         sz = size(F);
         x = 1:sz(2);
         y = 1:sz(1);

         for off_x = -(Nx-1):(Nx-1)
            ok_x = (x+off_x)>=1 & (x+off_x)<=sz(2);

            for off_y = -(Ny-1):(Ny-1)
               ok_y = (y+off_y)>=1 & (y+off_y)<=sz(1);
            
               nok = sum(ok_x)*sum(ok_y);
               F0 = reshape(F(y(ok_y), x(ok_x), :), nok, Nf);
               Foff = reshape(F(y(ok_y)+off_y, x(ok_x)+off_x, :), nok, Nf);

               im_cov_cell(off_y + Ny, off_x + Nx, :, :) = ...
                  im_cov_cell(off_y + Ny, off_x + Nx, :, :) + reshape(F0'*Foff, [1 1 Nf Nf]);
   
               im_counts(off_y + Ny, off_x + Nx) =  ...
                  im_counts(off_y + Ny, off_x + Nx) + nok;        
            end
         end
      end
   end
   
   im_cov_cell = bsxfun(@rdivide, im_cov_cell, im_counts);

   cov_cell = cov_cell + im_cov_cell;
end


cov_cell = cov_cell/length(ids);

mkdir('data');
save('data/cov_cell.mat', 'cov_cell');
