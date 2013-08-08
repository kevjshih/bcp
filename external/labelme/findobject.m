function  jc = findobject(fields, query, method)
% Utility function.
%
% fields is a cell array
% query is produced by query = parseContent(content);
% This function is called from several functions in the toolbox.
%
% You are probably looking for the function 'LMobjectindex'
%
% Methods:
%   [] = default = substring matching (air => chair)
%   'exact' = strings should match exactly
%   'word'  = the field should match the beggining (car => cars, air ~=> chair)

if nargin < 3
    method = '';
end

exactFlag = 0;
if (nargin == 3)
    if strcmp(lower(method), 'exact')
        exactFlag = 1;
    end
end

method = lower(method);

% Loop on all the objects in the current image.
Nobjects = length(fields); jc = [];
for m = 1:Nobjects
    f{m}=lower(strtrim(char(fields{m})));

    % Find the elements that match the query and that are not deleted
    queryresult = logical(0);

    if length(f{m})>0
        if strcmp(method, 'word')
            w = getwords(f{m});
        end

        % this is an OR loop for all the possible queries.
        for n = 1:length(query)
            Q = (strtrim(query{n}));
            found = logical(1);
            % this is an AND loop for all the words that have to be part of the
            % content of the field.
            for q = 1:length(Q)
                switch method
                    case '<'
                        try
                            % for dates:
                            j = datenum(f{m})<datenum(Q{q}(2:end));
                        catch
                            % for other strings:
                            j = f{m}<Q{q}(2:end); j = j(1);
                        end
                    case '>'
                        try
                            % for dates:
                            j = datenum(f{m})>datenum(Q{q}(2:end));
                        catch
                            % for other strings:
                            j = f{m}<Q{q}(2:end); j = j(1);
                        end
                    case 'exact'
                        % the field content must match the search exactly
                        j = strcmp(f{m}, Q{q}(2:end));
                    case 'word'
                        % search for exact words
                        j = strmatch(Q{q}(2:end), w, 'exact');
                    otherwise
                        % here just look if the chain is anywhere in the field
                        j = strfind(f{m}, Q{q}(2:end));
                end

                if j==0;
                    j = [];
                end

                if Q{q}(1)=='+'
                    found = found & (length(j)>0);
                else
                    found = found & (length(j)==0);
                end
            end
            queryresult = queryresult | found;
        end
    end
    jc(m) = queryresult;
end
jc = find(jc);

