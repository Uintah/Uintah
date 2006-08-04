function print_level_info()
%PRINT_LEVEL_MOVEMENT Print levels' information.
%   PRINT_LEVEL_MOVEMENT prints information of all levels at the current
%   timestep (number of patches, size of patches, size of cells, size of lattice,
%   etc.).
%   
%   See also TEST_MOVEMENT.

% Author: Oren Livne
%         06/28/2004    Added comments.

global_params;

o=1;
fprintf('Information of Levels\n\n');
fprintf('Level      Cell Size              Cell Num          Patch Size         Patch Num         Cell Ratio         Patch Ratio\n');
fprintf('-------------------------------------------------------------------------------------------------------------------------\n');
for k = o:length(L)+o-1
    fprintf('%2d   ',k);
    print_vector(L{k}.cell_size,'x','float');
    fprintf('    ');
    print_vector(L{k}.cell_num,'x','int');
    fprintf('    ');
    print_vector(L{k}.patch_size,'x','int');
    fprintf('    ');
    print_vector(L{k}.patch_num,'x','int');
    if (k < length(L)+o-1)
        fprintf('    ');
        print_vector(L{k}.rat_cell,'x','int');
        fprintf('    ');
        print_vector(L{k}.rat_patch,'x','int');
    end
    fprintf('\n');
end
fprintf('\n');
fprintf('Active patches:\n');
for k = o:length(L)+o-1
    fprintf('Level %d: #patches = %d\n',k,size(L{k}.patch_active,1));
    for p = o:size(L{k}.patch_active,1)+o-1,
        fprintf('\tPatch %3d, patch coords ',p);
        print_vector(L{k}.patch_active(p,:),',','int');
        fprintf('    start = ');
        print_vector(patch_start(k,L{k}.patch_active(p,:)));
        fprintf('  finish = ');
        print_vector(patch_finish(k,L{k}.patch_active(p,:)));
        fprintf('\n');
    end
end
fprintf('\n');
