function indexC = coarsenIndex(grid,k,indexF)

if (k == 1)
    error('Cannot coarsen index from coarsest level');
end

r = grid.level{k}.refRatio;
indexC = ceil(indexF./r);
