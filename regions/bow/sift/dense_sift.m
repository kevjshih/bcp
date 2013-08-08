function [map locs] = dense_sift(im, step)

if(~exist('step','var'))
    step = 1;
end
VLFEAT = 1;
SURF=0;

if(VLFEAT)
    %im = mean(im2single(im),3);
    
    im = im2single(im);
    im = mean(im,3);
    %im = rgb2hsv(im);
    nch = size(im,3);
    
    
    levels = [1 2 3 4];
    %levels = [3];
    levels = 1;
    
    %mask_inds = [41:56, 73:88];
    mask_inds = 1:128;
    
    for i = 1%2%1:levels
        level = levels(i);
        mapt = [];
        for c = 1:nch
%              [loc mapt2] = vl_dsift_resc(im(:,:,c), 'size', 2^levels(i),'fast', 'FloatDescriptors', 'Step',step);
              [loc mapt2] = vl_dsift(im(:,:,c), 'size', 2^levels(i),'fast', 'FloatDescriptors', 'Step',step);
              mapt = cat(1, mapt, mapt2(mask_inds,:));
        end
    
        if(step<=2)
            map{i} = double(reshape(mapt', [(max(loc(2,:)) - min(loc(2,:)))/step+1, (max(loc(1,:) - min(loc(1,:)))/step + 1), size(mapt,1)]));
            clear mapt
            if(nargout>=2)
            locs{i} = false(size(im,1), size(im,2));
            locs{i}(sub2ind([size(im,1), size(im,2)], loc(2,:), loc(1,:))) = 1;
            end
            
            % Contrast normalize
            len = 1;%sqrt(sum(map{i}.^2, 3));
            %map{i} = bsxfun(@rdivide, map{i}, len);
            %map{i}(isnan(map{i})) = 1/sqrt(size(map{i},3));
        else
            map{i} = double(mapt);
            locs{i} = loc;
            
            len = 1;%255;%sqrt(sum(map{i}.^2, 1));
            map{i} = bsxfun(@rdivide, map{i}, len);
            %map{i}(isnan(map{i})) = 1/sqrt(size(map{i},1));
        end
    end
elseif(SURF)
    
    %im = mean(im2single(im),3);
    im = im2double(im);
    im = vl_imsmooth(im, 2);
    im = rgb2hsv(im);
    
    
    nch = size(im,3);
    
    %mask_inds = [41:56, 73:88];
    
    for i = 1%2%1:levels
        mapt = [];
        for c = 1:nch
              %[loc mapt2] = vl_dsift(im(:,:,c), 'size', 2^levels(i),'fast', 'FloatDescriptors', 'Step',step);
              [mapt2 loc] = DenseSurf(im(:,:,c), 1, step, 2);
              mapt = cat(2, mapt, mapt2);
        end
    
        if(step==1)
            map = reshape(mapt, [loc.n, loc.m, size(mapt,2)]);%[max(loc(2,:)) - min(loc(2,:))+1, max(loc(1,:) - min(loc(1,:)) + 1), size(mapt,1)]);
            clear mapt
            
            % Pad map, apparently by 2
            % Contrast normalize
            %len = sqrt(sum(map{i}.^2, 3));
            %map{i} = bsxfun(@rdivide, map{i}, len);
            %map{i}(isnan(map{i})) = 1/sqrt(size(map{i},3));
        else
            map{i} = mapt';
            
            %len = sqrt(sum(map{i}.^2, 1));
            %map{i} = bsxfun(@rdivide, map{i}, len);
            %map{i}(isnan(map{i})) = 1/sqrt(size(map{i},1));
        end
    end
else % use siftflow implementation
    im = mean(im,3);
            
    levels = [1 2 3 4];
    i = 1;
    
    mask_inds = [41:56, 73:88];
    mapt = double(mexDenseSIFT(im, 2^levels(i), 1));
    sz = size(mapt);
    if(step<1)
        mapt = reshape(mapt, [], 128);
        r = randperm(size(mapt,1));
        map{i} = mapt(r(1:ceil(end*step)), mask_inds)';
        locs{i} = zeros(sz(1:2));
        locs{i}(sub2ind([numel(locs{i}),1], r(1:ceil(end*step)))) = 1;
    else
        map{i} = mapt(:,:,mask_inds);
    end
    
    locs{i} = ones(size(map{1},1), size(map{i},2));
end
