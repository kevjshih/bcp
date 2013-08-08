function [g_hog_blocks num_blocks] =hog2features(hog_c, patch_dims)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%
%%% hog2features: Converts HOG cells to features.
%%% For speed, the result is returned in the global variable g_hog_blocks
%%%
%%% Copyright (C) 2009, Lubomir Bourdev and Jitendra Malik.
%%% This code is distributed with a non-commercial research license.
%%% Please see the license file license.txt included in the source directory.
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load_config;
%global g_hog_blocks;
if ~config.USE_PHOG
   [num_blocks g_hog_blocks] =hog2features_internal(hog_c, patch_dims);
else
   error('Not prepared for this!\n');
end

end

function [num_blocks,g_hog_blocks]=hog2features_internal(hog_c,patch_dims)
load_config;

g_hog_blocks = [];

cell_size = config.HOG_CELL_DIMS./config.NUM_HOG_BINS;
block_size = patch_dims(2:-1:1)./cell_size(1:2);
hog_block_size = block_size-1;

[H,W,hog_hog_len] = size(hog_c);

num_blocks = max(0,[W H] - hog_block_size + 1);

block_hog_len = hog_hog_len*prod(hog_block_size);

% String them into blocks
num_num_blocks = prod(num_blocks);
if size(g_hog_blocks,1)<num_num_blocks
    g_hog_blocks = zeros(num_num_blocks,block_hog_len,'single');
elseif size(g_hog_blocks,1)>num_num_blocks
    g_hog_blocks((num_num_blocks+1):end,:) = [];
end

if num_num_blocks>0    
    for x=0:hog_block_size(1)-1
       for y=0:hog_block_size(2)-1
            g_hog_blocks(:,(x*hog_block_size(2)+y)*hog_hog_len+(1:hog_hog_len)) = reshape(hog_c(y+(1:num_blocks(2)),x+(1:num_blocks(1)),:),[num_num_blocks hog_hog_len]);
       end
    end

    % Reference implementation. Slow but more readable version that must
    % produce the same result as the real one
    if 0
        hog_blocks1 = zeros(num_num_blocks,block_hog_len,'single');
        for x=0:num_blocks(1)-1
            for y=0:num_blocks(2)-1
              hog_features=[];
              for xx=0:hog_block_size(1)-1
                   for yy=0:hog_block_size(2)-1
                        hog_features = [hog_features reshape(hog_c(y+yy+1,x+xx+1,:),[1 hog_hog_len])]; %#ok<AGROW>
                   end
              end
              hog_blocks1(x*num_blocks(2)+y+1,:) = hog_features;
            end
        end
        assert(isequal(g_hog_blocks,hog_blocks1));
    end    
end
end
