function [mag dI_dx dI_dy orient] = get_grad2(im)

if(size(im,3)>1)
   im = mean(im,3);
end

dI_dx = filter2([-1 0 1],im); 
dI_dy = filter2([-1 0 1]',im); 

mag = sqrt(dI_dx.^2 + dI_dy.^2);

if(nargout>3)
   orient = atan2(dI_dy, dI_dx); 
end
