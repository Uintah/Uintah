function grid = updateGrid(grid)
%UPDATEGRID  Update AMR grid tree.
%   After adding or deleting a patch, this function is called to update
%   index and subscript offsets of each patch, and update the cellIndex
%   maps of each patch.
%
%   See also: ADDGRIDPATCH.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

out(2,'--- Updating grid ---\n');

index = 0;
for k = 1:grid.numLevels,
    for q = 1:grid.level{k}.numPatches,
        P           = grid.level{k}.patch{q};

        % Update base index
        P.offsetInd = index;

        % Create cell index map (P.cellIndex)
        sub         = cell(grid.dim,1);
        for d = 1:grid.dim
            sub{d}  = [P.ilower(d)-1:P.iupper(d)+1] + P.offsetSub(d);
        end
        matSub      = cell(grid.dim,1);
        [matSub{:}] = myndgrid(sub{:});
        P.cellIndex = mysub2ind(P.size,matSub{:}) + P.offsetInd;

        grid.level{k}.patch{q} = P;
        index       = index + prod(P.size);

        % Create list of neighbouring patches of this patch
        for r = 1:grid.level{k}.numPatches,
            if (q == r)
                continue;
            end
            R = grid.level{k}.patch{r};
            for d = 1:grid.dim,
                if (P.ilower(d) == R.iupper(d)+1)
                    for other = setdiff(1:grid.dim,d)
                        if (    max(P.ilower(other),R.ilower(other)) <= ...
                                min(P.iupper(other),R.iupper(other)))
                            P.nbhrPatch(d,1) = r;
                            R.nbhrPatch(d,2) = q;
                        end
                    end
                end
                if (P.iupper(d) == R.ilower(d)-1)
                    for other = setdiff(1:grid.dim,d)
                        if (    max(P.ilower(other),R.ilower(other)) <= ...
                                min(P.iupper(other),R.iupper(other)))
                            P.nbhrPatch(d,2) = r;
                            R.nbhrPatch(d,1) = q;
                        end
                    end
                end
            end
            grid.level{k}.patch{q} = P;
            grid.level{k}.patch{r} = R;
        end
    end
end
grid.totalVars      = index;
