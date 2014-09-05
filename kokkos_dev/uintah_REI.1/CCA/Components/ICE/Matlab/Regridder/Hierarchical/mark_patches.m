function mark_patches(k)
%MARCH_PATCHES Mark patches for refinement.
%   MARK_PATCHES(K) marks patches for refinement at level K, based on the flagged
%   cells that are stored at L{K} based on the information from TOUT and object
%   rendering (movement) information.
%
%   See also OBJECT_RENDER, TEST_MOVEMENT.

% Author: Oren Livne
%         06/28/2004    Added comments.

global_params;

dim                 = length(L{k}.cell_num);                    % Dimension of the problem
b                   = box_list(zeros(1,dim),L{k}.rat_patch);    % Prepare all combinations for sub-patch index offsets
patch_new           = zeros(0,dim);                             % List of newly created patches at k+1
L{k}.deleted_patches= zeros(0,dim);
siz_sub             = L{k+1}.patch_size./L{k}.rat_cell;                 % Size of a sub-patch [L{k} cells]

for p = o:size(L{k}.patch_active,1)+o-1,                        % Loop over patches at this level
    j       = L{k}.patch_active(p,:);                           % d-D coordinates of the patch
    start   = patch_start(k,j);                                 % Start cell of patch j
    finish  = patch_finish(k,j);                                % Finish cell of patch j
    siz     = finish - start + 1;                               % Size of patch
    
    %%%%% Check whether there are flagged cells in each quadrant, create/delete patches on L{k+1} accordingly
    for q = o:size(b,1)+o-1,                                    % Loop over all possible patches; that might be 3x3 for a "normal" patch and 4x4 for a "big" last patch. So exit if we're outside the range of the L{k}-patch
        start_sub   = start + b(q,:).*siz_sub;                  % Sub-patch start cell [L{k} cells]
        if (~isempty(find(start_sub > finish)))                 % We are outside the L{k}-patch, skip this one
            continue;
        end
        start_f     = convert_c2f(k,start_sub,'cell');          % Sub-patch start cell [L{k+1} cells]
        j_f         = convert_c2p(k+1,start_f);                 % Sub-patch patch coordinate [L{k+1} patches]
        finish_sub  = start_sub + siz_sub - 1;
        last        = find(j_f == L{k+1}.patch_num);
        finish_sub(last) = L{k}.cell_num(last);
        ind         = patch_find(k+1,j_f);                      % Index of sub-patch at level k+1 (if exists)
        a           = find(check_range(...
            L{k}.cell_err_create,start_sub,finish_sub) > 0);    % Flagged cells in this patch (including small dilation), for creating patches
        ad          = find(check_range(...
            L{k}.cell_err_delete,start_sub,finish_sub) > 0);    % Flagged cells in this patch (including big dilation), for deleting patches
        if (~isempty(a) & (ind < 0))                            % There exist flagged cells of small dilation area and child doesn't exist => create it on L{k+1}
            if (tin.print >= 2)
                fprintf('-CREATE-  Patch [ ');
                fprintf('%d ',j);
                fprintf('], sub-patch [ ');
                fprintf('%d ',b(q,:));
                fprintf('] has flagged cells => refined to Level %d, Patch [ ',k+1);
                fprintf('%d ',j_f);
                fprintf(']\n');
            end
            L{k+1}.patch_active = [L{k+1}.patch_active; j_f];   % Add to level k+1 patches
            patch_new = [patch_new; j_f];                       % Add to list of newly created patches
            L{k+1}.num_created = L{k+1}.num_created+1;
        end
        if (isempty(ad) & (ind >= o))                           % No flagged cells of big dilation area and child exists => delete child and everything below it
            if (tin.print >= 2)
                fprintf('-DELETE-  Patch [ ');
                fprintf('%d ',j);
                fprintf('], quadrant %d has no flagged cells => delete Patch %d,%d [ ',q,k+1,ind);
                fprintf('%d ',j_f);
                fprintf('] + childs\n');
            end
            start_f     = j_f;
            finish_f    = j_f+1;            
            for l = k+1:length(L)+o-1
                c = L{l}.patch_active;
                a = find(check_range(c,start_f,finish_f-1) > 0);                
                if (tin.print >= 2)
                    fprintf('\tLevel %d: start_f=(%d,%d), finish_f=(%d,%d), children indices=',l,start_f,finish_f);
                    if (size(a,1) > 1)
                        a = a';
                    end
                    fprintf('%d ',a);
                    fprintf('\n');
                end
                if (isempty(a))
                    break;
                end
                L{l}.deleted_patches = [L{l}.deleted_patches; L{l}.patch_active(a,:)];  % Save deleted patches for plots
                L{l}.patch_active(a,:)  = [];                   % Delete patch children from level k+1 children list
                L{l}.num_deleted = L{l}.num_deleted + length(a);
                
                if (l < length(L)+o-1)
                    start_f  = convert_c2f(l,start_f,'patch');  % Convert from L{l-1} to L{l} coordinate
                    finish_f = convert_c2f(l,finish_f,'patch');
                end
            end
        end
    end
end

%%%%% Add cells at coarser levels to ensure that L{k+1} patches have safety layers from the
%%%%% boundaries of patches at this levels.
for l = k+1:-1:o+1,                                             % Loop from finest to coarsest
    patch_new   = L{l}.patch_active;
    if (isempty(patch_new))                                     % Until there are no patches to be added
        break;
    end
    patch_new_cells = zeros(0,dim);                             % Mark ALL L{l-1}-cells within the new L{l} patches
    for i = 1:size(patch_new,1),
        pf          = patch_new(i,:);                           % L{l} patch coords
        start       = convert_f2c(l,patch_start(l,pf),'cell');  % Starting cell of pf in L{l-1} cell coordinates
        finish      = convert_f2c(l,patch_finish(l,pf),'cell'); % Finishing cell of pf in L{l-1} cell coordinates    
        patch_new_cells = union(patch_new_cells,box_list(start,finish),'rows');
    end
    patch_new_cells_d = dilate_list(L{l-1}.cell_num,patch_new_cells,...
        tin.safe_bdry,'box');                                   % Dilate these cells by the #safety L{l-1}-cell layers we want; diagonal nbhrs count ('box')
    patch_needed    = unique(convert_c2p(l-1,patch_new_cells_d),'rows');         % L{l-1} patches that cover the L{l-1}-cells of interest
    add             = find(patch_find(l-1,patch_needed) < 0);   % Need to add these patches to L{l-1} because they don't exist
    patch_new       = patch_needed(add,:);                      % Added patches L{l-1} coordinates; update patch_new for yet-coarser level (l-1->l-2) update
    L{l-1}.patch_active = [L{l-1}.patch_active; patch_new];     % Add to level l-1        
        
    L{l-1}.new_patch        = patch_new;                        % Save added patches
    L{l-1}.new_cells        = patch_new_cells;                  % Save cells in new patches
    L{l-1}.new_cells_bdry   = patch_new_cells_d;                % Save dilated cells from new patches
    
    new     = size(patch_new,1);
    rein    = length(find(ismember(L{l-1}.deleted_patches,patch_new,'rows')));         % Reincarnated patches (deleted and then created => they were in fact untouched)
    L{l-1}.num_created  = L{l-1}.num_created + new - rein;
    L{l-1}.num_deleted  = L{l-1}.num_deleted - rein;
    
    if (~isempty(add))                                          % Print info on patches added to L{l-1}        
        if (tin.print >= 2)
            fprintf('-SAFE CREATE-  Level %d\n',l-1);
            for i = 1:size(patch_new,1),
                fprintf('\tPatch ');
                print_vector(patch_new(i,:),',','int');
                fprintf(' starts at ');
                print_vector(patch_start(k,patch_new(i,:)));
                fprintf('\n');
                
            end
        end
    end
end
