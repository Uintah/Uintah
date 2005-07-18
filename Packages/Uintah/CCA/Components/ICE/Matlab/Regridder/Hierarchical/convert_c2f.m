function f = convert_c2f(k,c,type)
%CONVERT_C2F Convert coarse level to fine level coordinates.
%   F = CONVERT_C2F(K,C,'CELL') converts L{k} cell coordinate C into its corresponding
%   L{k+1} coordinate F (i.e., F is the coordinate of the lower-left corner sub-cell
%   of L{k+1} contained in level K cell C. If C is a kxd array of coordinates, F will
%   be the corresponding kxd array of fine coordinates.
%   F = CONVERT_C2F(K,C,'PATCH') performs the above coordinate conversion of patch
%   coordinates (i.e. coordinate conversion w.r.t. lattices of levels K,K+1).
%   
%   See also CONVERT_F2C, CREATE_LEVELS.

% Author: Oren Livne
%         06/21/2004    Created and added comments.

global_params;

switch (type)
case 'cell',
    rat = L{k}.rat_cell;
case 'patch',
    rat = L{k}.rat_patch;
otherwise,
    error('bad type');
end

f   = repmat(rat,size(c,1),1).*(c-o) + o;
