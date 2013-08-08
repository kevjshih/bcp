function [D, unmatched] = LMaddtags(D, tagsfile, method)
%
% [D, unmatched] = LMaddtags(D, 'tags.txt');
%
% Reemplaces object names with the names in the list tags.txt
% The objects that are not matched are replaced by the label 'unmatched'
%
% The old labelme description will be stored in a new field: 
%  D.annotation.object.description
%
% method: specifies the replacements method'
%  'unmatched': it will add this tag is an object is not found (this is the
%  default)
%  'keepname': it will keep the old name for unmatched objects.

if nargin < 3
    method = 'unmatched';
end

[Tag, Descriptions] = loadtags(tagsfile);
Ntags = length(Tag);

% 1) Create list of labelme descriptions
[labelmeDescriptions, counts, imagendx, objectndx, descriptionndx] = LMobjectnames(D, 'name');

% 2) Find tag for each description and make list of unmatched descriptions
ndxtag = zeros(length(labelmeDescriptions),1);
for i = 1:length(labelmeDescriptions)
    for k = 1:Ntags
        j = strmatch(lower(labelmeDescriptions{i}), Descriptions{k}, 'exact');
        if ~isempty(j)
            ndxtag(i) = k;
            break
        end
    end
end

% Create list of unmatched descriptions and sort them by counts
unmatched = labelmeDescriptions(ndxtag==0);
[cc,jj]   = sort(counts(ndxtag==0), 'descend');
unmatched = unmatched(jj);

% 3) Add tag field to matched objects
for i = find(ndxtag>0)'
    j = find(descriptionndx==i);
    for k = 1:length(j)
        D(imagendx(j(k))).annotation.object(objectndx(j(k))).tag = Tag{ndxtag(i)};
        D(imagendx(j(k))).annotation.object(objectndx(j(k))).description = D(imagendx(j(k))).annotation.object(objectndx(j(k))).name;
        D(imagendx(j(k))).annotation.object(objectndx(j(k))).name = Tag{ndxtag(i)};
    end
end

% 4) Add unmatched tag for descriptions not matched
if strcmp(method, 'unmatched')
    for i = find(ndxtag==0)'
        j = find(descriptionndx==i);
        for k = 1:length(j)
            D(imagendx(j(k))).annotation.object(objectndx(j(k))).name = 'unmatched';
        end
    end
end

% Visualization
if nargout == 0
    [labelmetags, counttags] = LMobjectnames(D, 'tag');

    figure
    loglog(sort(counts, 'descend'), 'r')
    hold on
    loglog(sort(counttags(2:end), 'descend'), 'g')
    xlabel('rank')
    ylabel('counts')
    axis('tight')
    title(sprintf('descriptions: %d, Ntags:%d, polygons: %d, with tags: %d', length(labelmeDescriptions), Ntags, sum(counts), sum(counttags(2:end))))
end
