function showGist(gist, param)
%
% Visualization of the gist descriptor
%   showGist(gist, param)
%
% Example:
%   img = zeros(256,256);
%   img(64:128,64:128) = 255;
%   gist = LMgist(img, '', param);
%   showGist(gist, param)


[Nimages, Ndim] = size(gist);
nx = ceil(sqrt(Nimages)); ny = ceil(Nimages/nx);

Nblocks = param.numberBlocks;
Nfilters = sum(param.orientationsPerScale);
[ncols nrows Nfilters] = size(param.G);
Nfeatures = Nblocks^2*Nfilters;

if Ndim~=Nfeatures
    error('Missmatch between gist descriptors and the parameters');
end

G = param.G(1:2:end,1:2:end,:);
[ncols nrows Nfilters] = size(G);
G = G + flipdim(flipdim(G,1),2);
G = reshape(G, [ncols*nrows Nfilters]);


if Nimages>1
    figure;
end

for j = 1:Nimages
    g = reshape(gist(j,:), [Nblocks Nblocks Nfilters]);
    g = permute(g,[2 1 3]);
    g = reshape(g, [Nblocks*Nblocks Nfilters]);
        
    redg = g(:,1:Nfilters/2);
    greeng = g(:,Nfilters/2+1:Nfilters);

    redmosaic = reshape(G(:,1:Nfilters/2)*redg', [ncols nrows 1 Nblocks*Nblocks]);
    greenmosaic = reshape(G(:,Nfilters/2+1:Nfilters)*greeng', [ncols nrows 1 Nblocks*Nblocks]);

    mosaic = cat(3, redmosaic, greenmosaic, (redmosaic+greenmosaic)/4);
    %mosaic = reshape(G*g', [ncols nrows 1 Nblocks*Nblocks]);    
    mosaic = fftshift(fftshift(mosaic,1),2);
    mosaic = uint8(mosaic/max(mosaic(:))*255);
    
    if Nimages>1
        subplottight(ny,nx,j,0.01);
    end
    montage(mosaic)
end


