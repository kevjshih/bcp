function saveas_tight(h, varargin)
% Thanks to http://tipstrickshowtos.blogspot.com/2010/08/how-to-get-rid-of-white-margin-in.html


figure(h);

if(~exist('tight_type', 'var'))
    tight_type = 'TightInset';
end

switch(tight_type) 
    case 'TightInset' 
        ti = get(gca,'TightInset')
        
        set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
        set(gca,'units','centimeters')
        pos = get(gca,'Position');
        ti = get(gca,'TightInset');

        set(gcf, 'PaperUnits','centimeters');
        set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) pos(4)+ti(2)+ti(4)]);
    case 'Position' % not implemented right now
        ti = get(gca,'TightInset')
        
        set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);
        
        set(gca,'units','centimeters')
        pos = get(gca,'Position');

        set(gcf, 'PaperUnits','centimeters');
        set(gcf, 'PaperSize', [pos(3) pos(4)]);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperPosition',[0 0 pos(3) pos(4)]);
            
end

saveas(h, varargin{:});
