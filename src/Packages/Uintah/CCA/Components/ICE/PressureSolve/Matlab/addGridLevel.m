function [grid,k] = addGridLevel(grid,type,arg)
% Add a level to grid hierarchy; level id returned = k

grid.numLevels = grid.numLevels+1;
k = grid.numLevels;

if (k > grid.maxLevels)
    error('Maximum number of levels exceeded');
end

switch lower(type)   % LHS matrix, data is LHS matrix data structure
    
    case 'meshsize'
        grid.level{k}.h = arg;
        fprintf('Created empty level k=%d, meshsize = [%.5f %.5f]\n',k,arg);
        
    case 'refineratio',
        
        if (k == 1)
            error('Cannot create base level from a refinement ratio; use addGridLevel(...,''meshsize'',...)');
        end
        
        grid.level{k}.h = grid.level{k-1}.h ./ arg;
        fprintf('Created empty level k=%d, meshsize = [%.5f %.5f], refine ratio = [%3d %3d]\n',k,grid.level{k}.h,arg);                

    otherwise,
        
        error('addGridLevel: unknown type');
end

grid.level{k}.patch = cell(grid.maxPatches,1);
grid.level{k}.numPatches = 0;
