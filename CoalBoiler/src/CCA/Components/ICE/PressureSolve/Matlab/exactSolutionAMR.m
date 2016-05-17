function u = exactSolutionAMR(grid,T,TI)
%EXACTSOLUTIONAMR  Compute exact solution on the AMR hierarchy.
%   U = EXACTSOLUTIONAMR(GRID,T,TI) returns the value of the exact solution
%   values at nodes of the patches of the grid hierarchy GRID, using the
%   transformations T and TI to transfer boundary/ghost values to solution
%   values.
%
%   See also: AMRTOSPARSE, SPARSETOAMR, TESTADAPTIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

u = cell(grid.numLevels,1);

for k = 1:grid.numLevels,
    level       = grid.level{k};
    h = level.h;
    u{k} = cell(level.numPatches,1);

    for q = 1:grid.level{k}.numPatches,
        P = level.patch{q};

        %=====================================================================
        % Prepare a list of all interior cells
        %=====================================================================
        boxSize     = P.iupper - P.ilower+3;
        [indBox,box,matBox] = indexBox(P,P.ilower,P.iupper);                % Indices whose equations are created and added to Alist below
        u{k}{q}     = zeros(size(matBox{1}));
        boxOffset   = P.offsetSub;
        map         = P.cellIndex;
        % Interior point patch-based subscripts
        interior    = cell(grid.dim,1);
        for dim = 1:grid.dim
            interior{dim} = [P.ilower(dim):P.iupper(dim)] + boxOffset(dim);     % Patch-based cell indices including ghosts
        end
        matInterior      = cell(grid.dim,1);
        [matInterior{:}] = myndgrid(interior{:});
        pindexInterior   = mysub2ind(boxSize,matInterior{:});                 % Patch-based cell indices - list
        pindexInterior  = pindexInterior(:);
        mapInterior     = map(pindexInterior);
        % Ghost point physical locations
        interiorLocation   = cell(grid.dim,1);
        for dim = 1:grid.dim                                    % Translate face to patch-based indices (from index in the INTERIOR matInterior)
            interiorLocation{dim} = (interior{dim} - boxOffset(dim) - 0.5)*h(dim);
        end
        matInteriorLocation = cell(grid.dim,1);
        [matInteriorLocation{:}]   = myndgrid(interiorLocation{:});

        %=====================================================================
        % Compute edge indices
        %=====================================================================
        % Domain edges
        edgeDomain        = cell(2,1);
        edgeDomain{1}     = level.minCell + boxOffset;                      % First cell next to left domain boundary - patch-based index
        edgeDomain{2}     = level.maxCell + boxOffset;                      % Last Cell next to right domain boundary - patch-based index
        % Patch edges
        edgePatch          = cell(2,1);
        edgePatch{1}       = P.ilower + boxOffset;                            % First cell next to left domain boundary - patch-based index
        edgePatch{2}       = P.iupper + boxOffset;                            % Last Cell next to right domain boundary - patch-based index

        %=====================================================================
        % Set exact solution in the interior of the patch
        %=====================================================================
        u{k}{q}(interior{:}) = exactSolution(matInteriorLocation);

        %=====================================================================
        % Set boundary conditions
        %==================================================================        
        face = cell(grid.dim,1);
        for d = 1:grid.dim,                                                 % Loop over dimensions of patch
            for side = 1:2                                                  % side=1 (left) and side=2 (right) in dimension d
                if (side == 1)
                    direction = -1;
                else
                    direction = 1;
                end

                % Domain boundary, set boundary conditions
%                [face{:}] = find(matInterior{d} == edgeDomain{side}(d));          % Interior cell indices near PATCH boundary
                [face{:}] = myind2sub(size(matInterior{d}),find(matInterior{d} == edgeDomain{side}(d)));
                if (~isempty(face{1}))
                    for dim = 1:grid.dim                                    % Translate face to patch-based indices (from index in the INTERIOR matInterior)
                        face{dim} = face{dim} + 1;
                    end

                    % Ghost point patch-based subscripts
                    ghost           = face;
                    ghost{d}        = face{d} + direction;                              % Ghost cell indices
                    ghost{d}        = ghost{d}(1);                                      % Because face is an myndgrid-like structure, it repeats the same value in ghost{d}, lengthghostnbhr{d}) times. So shrink it back to a scalar so that myndgrid and flux(...) return the correct size vectors. We need to do that only for dimension d as we are on a face which is grid.dim-1 dimensional object.
                    matGhost        = cell(grid.dim,1);
                    [matGhost{:}]   = myndgrid(ghost{:});

                    % Inward normal direction
                    inwardNormal = zeros(1,grid.dim);
                    inwardNormal(d) = -direction;

                    % Ghost point physical locations
                    ghostLocation   = cell(grid.dim,1);
                    for dim = 1:grid.dim                                                % Translate face to patch-based indices (from index in the INTERIOR matInterior)
                        ghostLocation{dim} = (ghost{dim} - boxOffset(dim) - 0.5 + 0.5*inwardNormal(dim))*h(dim);
                    end
                    matGhostLocation = cell(grid.dim,1);
                    [matGhostLocation{:}]   = myndgrid(ghostLocation{:});
                    u{k}{q}(ghost{:}) = exactSolution(matGhostLocation);
                    continue;                                                           % Skip the rest of the code in this loop as B.C. prevail on C/F interface values
                end

                % Interior C/F interface, set solution values
%                [face{:}] = find(matInterior{d} == edgePatch{side}(d));                 % Interior cell indices near PATCH boundary
                [face{:}] = myind2sub(size(matInterior{d}),find(matInterior{d} == edgePatch{side}(d)));
                if (~isempty(face{1}))
                    for dim = 1:grid.dim                                                % Translate face to patch-based indices (from index in the INTERIOR matInterior)
                        face{dim} = face{dim} + 1;
                    end

                    % Ghost point patch-based subscripts
                    ghost           = face;
                    ghost{d}        = face{d} + direction;                              % Ghost cell indices
                    ghost{d}        = ghost{d}(1);                                      % Because face is an myndgrid-like structure, it repeats the same value in ghost{d}, lengthghostnbhr{d}) times. So shrink it back to a scalar so that myndgrid and flux(...) return the correct size vectors. We need to do that only for dimension d as we are on a face which is grid.dim-1 dimensional object.
                    matGhost        = cell(grid.dim,1);
                    [matGhost{:}]   = myndgrid(ghost{:});

                    % Ghost point physical locations
                    ghostLocation   = cell(grid.dim,1);
                    for dim = 1:grid.dim                                                % Translate face to patch-based indices (from index in the INTERIOR matInterior)
                        ghostLocation{dim} = (ghost{dim} - boxOffset(dim) - 0.5)*h(dim);
                    end
                    matGhostLocation = cell(grid.dim,1);
                    [matGhostLocation{:}]   = myndgrid(ghostLocation{:});
                    u{k}{q}(ghost{:}) = exactSolution(matGhostLocation);
                end

            end
        end

%         %=====================================================================
%         % Delete values in deleted Boxes
%         %=====================================================================
%         % Interior point patch-based subscripts
%         for i = 1:length(P.deletedBoxes)
%             B = P.deletedBoxes{i};
%             box = cell(grid.dim,1);
%             for dim = 1:grid.dim
%                 box{dim} = [B.ilower(dim):B.iupper(dim)] + boxOffset(dim);     % Patch-based cell indices including ghosts
%             end
%             u{k}{q}(box{:}) = 0.0;
%         end
    end
end

% Affect zero values at unused gridpoints
temp    = AMRToSparse(u,grid,T,1);
u       = sparseToAMR(temp,grid,TI,1);
