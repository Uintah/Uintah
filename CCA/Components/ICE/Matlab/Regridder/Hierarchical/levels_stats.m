function s = levels_stats()
%LEVELS_STATS Compute statistics at all levels.
%   S = LEVELS_STATS computes statistics on the efficiency of patches, their number, etc.
%   at all levels, and stores it in the structure S (all this refers to a specific
%   timestep).
%
%   See also CREATE_LEVELS, FINAL_PLOTS, TEST_MOVEMENT.

% Author: Oren Livne
%         06/28/2004    Added comments.

global_params;

s                   = zeros(length(L),9);
dim                 = length(tin.domain);                           % Dimension of the problem

for k = o:length(L)+o-1,                                            % Loop over levels
    c                   = k-1;                                      % Coarse level index
    np                  = size(L{k}.patch_active,1);
    nc                  = L{k}.num_created;
    nd                  = L{k}.num_deleted;
    if (k == o)
        f               = [];
    else
        f               = L{c}.cell_err;                            % Flagged cells d-D indices in array form        
    end
    
    %%%%% Compute patch statistics
    eff                 = zeros(np,1);
    flagged             = zeros(np,1);
    volume              = zeros(np,1);    
    for p = o:np+o-1,
        j       = L{k}.patch_active(p,:);                           % d-D coordinates of the patch
        start   = patch_start(k,j);                                 % Start cell of patch j at level k
        finish  = patch_finish(k,j);                                % Start cell of diagonal nbhr of patch j at level k
        siz     = finish - start + 1;                               % Size of this patch            
        volume(p)  = prod(siz);
        if (k > o)
            start   = convert_f2c(k,start,'cell');                      % Start cell of patch j at level c
            finish  = convert_f2c(k,finish,'cell');                     % Start cell of diagonal nbhr of patch j at level c
            siz     = finish - start + 1;                               % Size of this patch
            flagged(p) = length(find(check_range(f,start,finish) > 0)); % Flagged cells in this patch        
            eff(p)  = flagged(p)/prod(siz);
        end
    end    
    if (np > 0)
        eff             = flagged./volume;                          % Patch efficiency
        s(k,:)          = [np sum(volume) np/prod(L{k}.patch_num) mean(eff) median(eff) max(eff) length(find(eff==0)) nc nd];
    else
        s(k,:)          = [np sum(volume) np/prod(L{k}.patch_num) 0 0 0 0 nc nd];            
    end
end
