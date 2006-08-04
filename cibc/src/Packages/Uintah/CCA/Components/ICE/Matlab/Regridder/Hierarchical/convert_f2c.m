function c = convert_f2c(k,f,type)
%CONVERT_F2C Convert fine level to coarse level coordinates.
%   C = CONVERT_F2C(K,F,'CELL') converts L{k} cell coordinate F into its corresponding
%   L{k-1} coordinate C (i.e., F is in the coarse cell C). If F is a kxd array of
%   coordinates, C will be the corresponding kxd array of fine coordinates.
%   F = CONVERT_F2C(K,F,'PATCH') performs the corresponding conversion of patch coordinates.
%   
%   See also CONVERT_C2F, CREATE_LEVELS.

% Author: Oren Livne
%         06/21/2004    Created and added comments.

global_params;

switch (type)
case 'cell',
    rat = L{k-1}.rat_cell;
    num = L{k-1}.cell_num;
case 'patch',
    rat = L{k-1}.rat_patch;
    num = L{k-1}.patch_num;
otherwise,
    error('bad type');
end

c               = floor((f-o)./repmat(rat,size(f,1),1)) + o;
last            = find(c > num);
c(last)         = num(last);
