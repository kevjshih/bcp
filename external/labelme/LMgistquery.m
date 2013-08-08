function j = LMgistquery(gistQuery, gist)

if 1
    % normalize correlation
    gistQuery = normalizeGist(gistQuery);
    gist = normalizeGist(gist);
    D = gist*gistQuery';
    [D,j] = sort(D, 'descend');
else
    % L2
    D = sum((gist - repmat(gistQuery, [size(gist,1) 1])).^2,2);
    [D,j] = sort(D);
end


