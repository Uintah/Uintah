function u = sparseToAMR(x,grid)
% Strip the long vector of composite grid variables x to patch-based data
% in u.
u = cell(grid.numLevels,1);

for k = 1:grid.numLevels,
    level       = grid.level{k};
    u{k} = cell(level.numPatches,1);

    for q = 1:grid.level{k}.numPatches,
        P = level.patch{q};

        %=====================================================================
        % Prepare a list of all interior cells
        %=====================================================================
        boxSize     = P.iupper - P.ilower+3;
        boxOffset   = P.offset;
        map         = P.cellIndex;
        interior    = cell(grid.dim,1);
        for dim = 1:grid.dim
            interior{dim} = [P.ilower(dim):P.iupper(dim)] + boxOffset(dim);     % Patch-based cell indices including ghosts
        end
        matInterior      = cell(grid.dim,1);
        [matInterior{:}] = ndgrid(interior{:});
        pindexInterior   = sub2ind(boxSize,matInterior{:});                 % Patch-based cell indices - list
        pindexInterior  = pindexInterior(:);
        mapInterior     = map(pindexInterior);

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

        % Restore boundary point values from fluxes
        face = cell(grid.dim,1);
        for d = 1:grid.dim,                                                 % Loop over dimensions of patch
            for side = 1:2                                                  % side=1 (left) and side=2 (right) in dimension d
                if (side == 1)
                    direction = -1;
                else
                    direction = 1;
                end
                [face{:}] = find(matInterior{d} == edgePatch{side}(d));          % Interior cell indices near PATCH boundary
                for dim = 1:grid.dim                                    % Translate face to patch-based indices (from index in the INTERIOR matInterior)
                    face{dim} = face{dim} + 1;
                end
                ghost           = face;
                ghost{d}        = face{d} + direction;                              % Ghost cell indices
                ghost{d}        = ghost{d}(1);                                      % Because face is an ndgrid-like structure, it repeats the same value in ghost{d}, lengthghostnbhr{d}) times. So shrink it back to a scalar so that ndgrid and flux(...) return the correct size vectors. We need to do that only for dimension d as we are on a face which is grid.dim-1 dimensional object.
                matGhost        = cell(grid.dim,1);
                [matGhost{:}]   = ndgrid(ghost{:});
                pindexGhost     = sub2ind(boxSize,matGhost{:});
                pindexGhost     = pindexGhost(:);
                mapGhost        = map(pindexGhost);

                % Inward normal direction
                nbhrOffset = zeros(1,grid.dim);
                nbhrOffset(d) = -direction;
                % Nbhrs of all ghost cells in this direction
                nbhr = cell(grid.dim,1);
                for dim = 1:grid.dim
                    nbhr{dim} = ghost{dim} + nbhrOffset(dim);
                end
                matNbhr         = cell(grid.dim,1);
                [matNbhr{:}]    = ndgrid(nbhr{:});
                pindexNbhr      = sub2ind(boxSize,matNbhr{:});                      % Patch-based cell indices - list
                pindexNbhr      = pindexNbhr(:);
                mapNbhr         = map(pindexNbhr);

                x(mapGhost) = x(mapGhost) + x(mapNbhr);
            end

        end

        u{k}{q} = x(P.cellIndex);

    end
end
