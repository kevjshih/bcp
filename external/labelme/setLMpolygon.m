function polygon = setLMpolygon(x,y)
  
  for i = 1:length(x)
    polygon.pt(i).x = num2str(x(i));
    polygon.pt(i).y = num2str(y(i));
  end
