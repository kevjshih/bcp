function m = init_clustered_model(aspect)

% Find window that has closest area to an 8x8 model
area = 64;
w = sqrt(area/aspect);
h = w*aspect;

width = round(w);
height = round(h);

w = ones(height, width, 31); % Encourages it to have high energy?

m.w = w;
m.size = size(w);
m.b = 0;
m.spat_const = [0 1 0 1 0.75 1]; % Object level
