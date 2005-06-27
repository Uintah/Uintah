function [grid,k] = addGridLevel(grid,type,arg)
global verboseLevel
% Add a level to grid hierarchy; level id returned = k

grid.numLevels = grid.numLevels+1;
k = grid.numLevels;

if (k > grid.maxLevels)
    error('Maximum number of levels exceeded');
end

switch lower(type)   % LHS matrix, data is LHS matrix data structure

    case 'meshsize'
        grid.level{k}.h = arg;
        if (k > 1)
            grid.level{k}.refRatio = grid.level{k-1}.h ./ arg;
            if (max(abs(mod(grid.level{k-1}.h,arg))) > eps)
                error('Cannot create a level with non-integer refinement ratio w.r.t. its coarser level');
            end
        end

    case 'refineratio',

        if (k == 1)
            error('Cannot create base level from a refinement ratio; use addGridLevel(...,''meshsize'',...)');
        end

        grid.level{k}.refRatio  = arg;
        grid.level{k}.h         = grid.level{k-1}.h ./ arg;

    otherwise,

        error('addGridLevel: unknown type');
end

if (verboseLevel >= 1)
    if (k == 1)
        fprintf('Created empty level k=%d, meshsize = [%.5f %.5f]\n',k,grid.level{k}.h);
    else
        fprintf('Created empty level k=%d, meshsize = [%.5f %.5f], refine ratio = [%3d %3d]\n',k,grid.level{k}.h,grid.level{k}.refRatio);
    end
end

grid.level{k}.domainSize    = grid.domainSize;
grid.level{k}.minCell       = ones(1,grid.dim);
grid.level{k}.maxCell       = grid.domainSize./grid.level{k}.h;

grid.level{k}.patch = cell(grid.maxPatches,1);
grid.level{k}.numPatches = 0;
