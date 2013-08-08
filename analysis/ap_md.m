function ap_out = VOCap(rec,prec)


% I'm lazy, just doing a for loop
ap_out = zeros(size(prec, 1), 1);

for i = 1:size(prec, 1)
   ap_out(i) = ap(rec(i, :)', prec(i, :)');
end


