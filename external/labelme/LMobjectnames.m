function [names, counts, imagendx, objectndx, descriptionndx] = LMobjectnames(D, field);
% Returns the name of all the object classes in the database, 
% and the number of instances of each object class:
%
% [names, counts] = LMobjectnames(D);
%
% You can visualize the counts and object names by calling the function
% without output arguments:
%
% LMobjectnames(D);
%
% You can see the list of words associated with an object class using the
% command LMobjectnames. Some examples:
%   LMobjectnames(LMquery(D, 'name', 'face'))
%   LMobjectnames(LMquery(D, 'name', 'plate'))
%   LMobjectnames(LMquery(D, 'name', 'person'))
%
% [names, counts, imagendx, objectndx, descriptionndx] = LMobjectnames(D);

if nargin == 1
    field = 'name';
end

Nannotations = length(D);
Npolygons = 20*Nannotations;

names = {};
imagendx = zeros(Npolygons,1); 
objectndx = zeros(Npolygons,1);

m = 0;

for i = 1:Nannotations
    if mod(i,100)==0; disp(i); end

    if isfield(D(i).annotation, 'object')
        if isfield(D(i).annotation.object, field)
            N = length(D(i).annotation.object);
            for j = 1:N
                %
                m = m+1;
                if length(D(i).annotation.object(j).(field))>0
                    names{m} = D(i).annotation.object(j).(field);
                else
                    names{m} = '';
                end
                imagendx(m) = i;
                objectndx(m) = j;
            end
        end
    end
end
imagendx = imagendx(1:m);
objectndx = objectndx(1:m);

[foo, i, descriptionndx] = unique(strtrim(lower(names)));
names = strtrim(names(i));

if nargout ~= 1 | nargin == 2
    Nobject = length(names);
    [counts, x] = hist(descriptionndx, 1:Nobject);
end


if nargout == 0
    % plot counts
    jj = sort(counts, 'descend');
    figure
    barh(counts(jj))
    set(gca, 'YTick', 1:Nobject)
    set(gca, 'YtickLabel', names(jj))
    axis([0 max(counts)+5 0 Nobject+1])
    grid on
end
