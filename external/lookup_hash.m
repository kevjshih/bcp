function results = lookup_hash(words, hash);
% results = lookup_hash(words, hash)
%
% Input:
%   words - string or N x 1 cell array of words to lookup
%   hash - hash table built with make_hash
%
% Output:
%   results - N x 1 array with index corresponding to each word in words

words = strrep(words, ' ', '_');
words = strrep(words, '.', '_');

if(~iscell(words))
   words = {words};
end

results = zeros(size(words));

for i = 1:numel(words)
   results(i) = hash.(words{i});
end
