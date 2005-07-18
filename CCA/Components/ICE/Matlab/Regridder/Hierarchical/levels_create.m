function levels_create(max_levels)
%LEVELS_CREATE Initialize levels.
%   LEVELS_CREATE(MAX_LEVELS) initializes the global cell array L according to the
%   global input structure TOUT. This includes cell number and size, patch number and
%   size, and various lists of active/recently created/recently deleted patches.
%
%   See also TEST_MOVEMENT.

% Author: Oren Livne
%         06/28/2004    Added comments.

global_params;

dim                 = length(tin.domain);                           % Dimension of the problem

for k = o:length(L)+o-1,                                            % Create levels from coarsest to finest
    if (k == o)                                                     % Coarsest level
        L{k}.cell_num       = tin.cell_num;                         % Size of grid [cell]        
        L{k}.patch_size     = tin.patch_size;                       % Patch size [cell]        
        L{k}.patch_num      = divide(L{k}.cell_num,L{k}.patch_size);% Number of patches [patch]
        L{k}.patch_active   = box_list(zeros(1,dim)+o,...
            L{k}.patch_num);                                        % List of active patches (all active)
    else
        c                   = k-1;                                  % Coarse level index
        L{k}.cell_num       = L{c}.cell_num.*L{c}.rat_cell;         % Size of this grid [cell], twice finer
        L{k}.patch_size     = L{c}.patch_size.*...
            L{c}.rat_cell./L{c}.rat_patch;                          % Patch size [cell], has to be integer
        if (~isempty(find(rem(L{k}.patch_size,1) ~= 0)))            % We assume that each "normal" patch is divided into even sub-patches - no "last sub-patches" are allowed
            s1 = sprintf('Bad input parameters of lattice/cell refinement ratio: level %d, non-integer patch size [',k);
            s2 = sprintf('%d ',L{k}.patch_size);
            s3 = sprintf(']');
            s = [s1 s2 s3];
            error(s);
        end
        L{k}.patch_num      = divide(L{k}.cell_num,L{k}.patch_size);% Number of patches [patch]
        L{k}.patch_active   = zeros(0,dim);                         % List of active patches (empty)
    end        
    L{k}.cell_size          = tin.domain./L{k}.cell_num;            % Meshsize [meter], for simplicity h=1 here            
    L{k}.cell_err           = zeros(0,dim);                         % List of flagged cells (empty)
    L{k}.cell_err_create    = zeros(0,dim);                         % Dilated List of flagged cells for creating patches
    L{k}.cell_err_delete    = zeros(0,dim);                         % Dilated List of flagged cells for deleting patches
    
    if (k < length(L)+o-1)
        L{k}.rat_cell   = tin.rat_cell(k,:);                        % Grid refinement ratio L{k}->L{k+1}
        L{k}.rat_patch  = tin.rat_patch(k,:);                       % lattice refinement ratio L{k}->L{k+1}
    end
    
    %%%%% The following variables are only needed for plots/stats
    L{k}.new_patch          = zeros(0,dim);                         % List of newly created patches (empty)
    L{k}.new_cells          = zeros(0,dim);                         % List of cells in new patches
    L{k}.new_cells_bdry     = zeros(0,dim);                         % List of dilated cells from new patches            
    L{k}.deleted_patches    = zeros(0,dim);                         % List of deleted patches (empty)
    L{k}.num_created    = 0;
    L{k}.num_deleted    = 0;
end
