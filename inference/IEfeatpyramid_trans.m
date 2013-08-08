function [feat, scale, sizes, scaled_rot0] = featpyramid(im, sbin, interval, shift, rotation)

% [feat, scale] = featpyramid(im, sbin, interval);
% Compute feature pyramid.
%
% sbin is the size of a HOG cell - it should be even.
% interval is the number of scales in an octave of the pyramid.
% feat{i} is the i-th level of the feature pyramid.
% scale{i} is the scaling factor used for the i-th level.
% feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

sc = 2 ^(1/interval);
imsize = [size(im, 1) size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));
feat = cell(max_scale, length(shift), length(shift), length(rotation));
scale = zeros(max_scale, 1);

%resize function wants floating point values
im = im2double(im);

tic;

scaled0 = im;
for i_rot = 1:length(rotation)
   rot = rotation(i_rot);
   if(rot==0)
      scaled_rot0{i_rot} = scaled0;
      mask_rot0{i_rot} = ones(size(scaled0, 1), size(scaled0, 2));
   else
      % Imrotate is at least twice as fast for uint8 ...
      scaled_rot0{i_rot} = im2double(imrotate(im2uint16(scaled0), rot));%, 'bilinear'));
      mask_rot0{i_rot} = double(imerode(imrotate(ones(size(scaled0,1), size(scaled0,2), 'uint8'), rot)==1, ones(sbin/2+1,sbin/2+1)));
   end
end


for i = 1:interval
    
   for i_rot = 1:length(rotation)
      scaled_rot{i_rot} = IEresize(scaled_rot0{i_rot}, 1/sc^(i-1));
      mask_rot{i_rot} = IEresize(mask_rot0{i_rot}, 1/sc^(i-1));
   end

   %%%%%%%%%%% Loop over all ocataves %%%%%%%%%%%%
   for j = i:interval:max_scale
      for i_rot = 1:length(rotation)
         % Setup Images and Masks
         if(j>i) % Don't downsample on lowest level
            scaled_rot{i_rot} = IEresize(scaled_rot{i_rot}, 0.5);
            mask_rot{i_rot} = IEresize(mask_rot{i_rot}, 0.5);
         end
         
         for i_x_sh = 1:length(shift)
            x_sh = shift(i_x_sh);
            for i_y_sh = 1:length(shift)
               y_sh = shift(i_y_sh);

               cur_im = scaled_rot{i_rot}(y_sh+1:end, x_sh+1:end, :);
               cur_mask = double(mask_rot{i_rot}(y_sh+1:end, x_sh+1:end)>0.5);
               
               feat{j, i_x_sh, i_y_sh, i_rot} = features_w(cur_im, cur_mask, sbin);
               sizes{j, i_x_sh, i_y_sh, i_rot} = size(cur_im);
            end
         end
      end

      scale(j) = 1/sc^(j-1);
   end
end

