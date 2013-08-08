function draw_bbox(box, varargin)
% draw_bbox([x_min y_min x_max y_max], plot params);

if(~isempty(box))
    plot([box(:,1) box(:,1) box(:,3) box(:,3), box(:,1)]', [box(:,2), box(:,4), box(:,4), box(:,2), box(:,2)]', varargin{:})
    
    if(size(box,2)>4)
        for i = 1:size(box,1)
            text(box(i,1), box(i,2), sprintf('%.3f', box(i,end)), 'backgroundcolor', 'k','color', 'w');
        end
    end
    
end