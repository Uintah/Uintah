function [As,bs] = AMRToSparse(A,b,grid)

Alist = zeros(0,3);
bs = zeros(36,1);
numLevels = length(grid);

for k = 1:numLevels,
    %k
    g = grid{k};
    s = g.stencilOffsets;
    numPatches = length(g.patch);
    numEntries = size(s,1);

    for p = 1:numPatches,
        P = g.patch{p};
        PSize = P.iupper-P.ilower+1;
        POffset     = -P.ilower+2;                   % Add to physical cell index to get patch cell index

        i1All = [P.ilower(1)-1:P.iupper(1)+1]+POffset(1);
        i2All = [P.ilower(2)-1:P.iupper(2)+1]+POffset(2);
        [mat1,mat2] = ndgrid(i1All,i2All);        
        diagIndexAll = sub2ind(PSize+2,mat1,mat2)+P.baseIndex-1;
        diagIndexAll = diagIndexAll(:);

        i1 = [P.ilower(1):P.iupper(1)]+POffset(1);
        i2 = [P.ilower(2):P.iupper(2)]+POffset(2);
        [mat1,mat2] = ndgrid(i1,i2);        
        diagIndex = sub2ind(PSize+2,mat1,mat2)+P.baseIndex-1;
        diagIndex = diagIndex(:);

        ghostIndex = setdiff(diagIndexAll,diagIndex);

        % Add stencil entries to A and b
        for entry = 1:numEntries,
            j1 = i1 + s(entry,1);
            j2 = i2 + s(entry,2);
            [mat1,mat2] = ndgrid(j1,j2);
            nbhrIndex = sub2ind(PSize+2,mat1,mat2)+P.baseIndex-1;
            nbhrIndex = nbhrIndex(:);
            stencilValues = A{k}{p}(i1,i2,entry);
            Alist = [Alist; [diagIndex nbhrIndex stencilValues(:)]];
        end
        rhsValues = b{k}{p}(i1,i2);
        bs(diagIndex) = rhsValues(:);

        % Add boundary equations to A and b
        % Dirichlet B.C. u=0
        Alist = [Alist; [ghostIndex ghostIndex repmat(1.0,size(ghostIndex))]];
        bs(ghostIndex) = 0.0;
        
    end


end

As = sparse(Alist(:,1),Alist(:,2),Alist(:,3));










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (0)
    numDims = length(ilower);
    gridSize = g.iend - g.istart + 1;

    i = cell(numDims,1);
    diag = cell(numDims,1);
    for d = 1:numDims,
        diag{d} = [ilower(d):iupper(d)] - g.istart(d) + 1;
        i{d} = diag{d} + s(d);
    end
    cells = cell(numDims,1);
    [cells{:}] = ndgrid(i{:});
    cellsDiag = cell(numDims,1);
    [cellsDiag{:}] = ndgrid(diag{:});

    numCells = length(cells{1}(:));
    cellsMat = zeros(numCells,numDims);
    cellsMatDiag = zeros(numCells,numDims);
    for d = 1:numDims,
        cellsMat(:,d) = cells{d}(:);
        cellsMatDiag(:,d) = cellsDiag{d}(:);
    end

    for d = 1:numDims,
        %d
        outside = find(max((cellsMat < repmat(ones(size(gridSize)),[size(cellsMat,1) 1])) | ...
            (cellsMat > repmat(gridSize,[size(cellsMat,1) 1])),[],2));
        if (~isempty(outside))
            cellsMat(outside,:) = [];
            cellsMatDiag(outside,:) = [];
            values(outside,:) = [];
        end
    end

    for d = 1:numDims,
        cells{d} = cellsMat(:,d);
        cellsDiag{d} = cellsMatDiag(:,d);
    end

    index = sub2ind(gridSize,cells{:});
    indexDiag = sub2ind(gridSize,cellsDiag{:});
    %indexDiag
    %length(indexDiag)

    A = [A; [indexDiag + g.iBase - 1 index + g.iBase - 1 values]];
end
