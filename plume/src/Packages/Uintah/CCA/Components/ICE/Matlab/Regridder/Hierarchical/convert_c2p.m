function p = convert_c2p(k,c)
%CONVERT_C2P Convert cell to patch coordinates.
%   P = CONVERT_C2P(K,C) converts L{k} cell coordinate C into its corresponding
%   patch coordinate P (i.e., (i.e. cell C is in patch P). If C is a kxd array of
%   cell coordinates, P will be the corresponding kxd array of patch coordinates.
%   
%   See also CONVERT_C2F, CONVERT_F2C, CREATE_LEVELS, PATCH_START.

% Author: Oren Livne
%         06/21/2004    Created and added comments.

global_params;

n = size(c,1);
if (n == 0)
    p = zeros(size(c));
else
    p       = floor((c-o)./repmat(L{k}.patch_size,n,1)) + o;
    num     = repmat(L{k}.patch_num,n,1);
    last    = find(p > num);
    p(last) = num(last);
end
