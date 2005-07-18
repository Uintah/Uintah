function ind = patch_find(k,pf)
%PATCH_FIND Return patch index from its coordinates at a given level.
%   IND = PATCH_FIND(K,PF) returns the 1D index of the patch specified by 
%   the patch d-D coordinate PF, at level K. This is done by a slow search,
%   can be done later in a faster way using a lexicographic ordering of
%   the coordinates in the list. If PF is a list of d-D coordinates, IND
%   is a vector of their indices.
%   
%   See also PATCH_FINISH, PATCH_START.

% Author: Oren Livne
%         06/28/2004    Version 1: Created

global_params;

ind = -ones(size(pf,1),1);

for i = o:size(pf,1)+o-1,
    temp = find(ismember(L{k}.patch_active,pf(i,:),'rows'));
    if (~isempty(temp))
        ind(i) = temp;
    end
end
