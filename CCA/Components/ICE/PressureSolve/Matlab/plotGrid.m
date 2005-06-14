function plotGrid(grid,fileName,showGhosts,showCellIndex)
% Plot gridlines (works in 2D only). If showGhosts=1 we add ghost cells to
% the plot.

if (nargin < 2)
    fileName = 'grid.eps';
end
if (nargin < 3)
    showGhosts = 0;
end
if (nargin < 4)
    showCellIndex = 0;
end

figure(1);
clf;

levelColor = {'blue','red','green'};

for k = 1:grid.numLevels,
    level = grid.level{k};
    h = level.h;
    if (k == 1)
        offsetLower = [-1.5 -1.5];
        offsetUpper = [0.5 0.5];
        if (showGhosts)
            offsetLower = offsetLower - 1;
            offsetUpper = offsetUpper + 1;
        end
        axis([...
            (level.minCell(1)+offsetLower(1))*h(1) (level.maxCell(1)+offsetUpper(1))*h(1) ...
            (level.minCell(2)+offsetLower(2))*h(2) (level.maxCell(2)+offsetUpper(2))*h(2) ...
            ]);
        hold on;
    end
    set(gcf,'position',[488   233   850   850]);

    for q = 1:grid.level{k}.numPatches,
        P = grid.level{k}.patch{q};

        i1          = [P.ilower(1)-1:P.iupper(1)+1];
        i2          = [P.ilower(2)-1:P.iupper(2)+1];
        [mat1,mat2] = ndgrid(i1,i2);
        x1          = (mat1-0.5)*h(1);
        x1          = x1(:);
        x2          = (mat2-0.5)*h(2);
        x2          = x2(:);

        for i = 1:length(x1)
            if (    (mat1(i) >= P.ilower(1)) & (mat1(i) <= P.iupper(1)) & ...
                    (mat2(i) >= P.ilower(2)) & (mat2(i) <= P.iupper(2)))
                a = rectangle('Position',[x1(i)-0.5*h(1) x2(i)-0.5*h(2) h(1) h(2)]);
                set(a,'edgecolor',levelColor{k});
                set(a,'linewidth',k);
                set(a,'linestyle','-')

                b = plot(x1(i),x2(i),'o');
                set(b,'markersize',10);
                set(b,'markeredgecolor',levelColor{k});
                set(b,'markerfacecolor',levelColor{k});

                if (showCellIndex)
                    index = P.cellIndex(mat1(i)+P.offset(1),mat2(i)+P.offset(2));
                    c = text(x1(i)-0.125*h(1),x2(i)-0.125*h(2),sprintf('%d',index));
                    set(c,'fontsize',10);
                    set(c,'color','k');
                end

            elseif (showGhosts)
                a = rectangle('Position',[x1(i)-0.5*h(1) x2(i)-0.5*h(2) h(1) h(2)]);
                set(a,'edgecolor',levelColor{k});
                set(a,'linewidth',k);
                set(a,'linestyle','--')

                b = plot(x1(i),x2(i),'o');
                set(b,'markersize',10);
                set(b,'markeredgecolor',levelColor{k});
                set(b,'markerfacecolor','white');

                if (showCellIndex)
                    index = P.cellIndex(mat1(i)+P.offset(1),mat2(i)+P.offset(2));
                    c = text(x1(i)-0.125*h(1),x2(i)-0.125*h(2),sprintf('%d',index));
                    set(c,'fontsize',10);
                    set(c,'color','k');
                end
            end
        end

        %         % Set boundary conditions
        %         for i1 = P.ilower(1)-1:P.iupper(1)+1
        %             for i2 = P.ilower(2)-1:P.iupper(2)+1
        %                 if (    (i1 >= P.ilower(1)) & (i1 <= P.iupper(1)) & ...
        %                         (i2 >= P.ilower(2)) & (i2 <= P.iupper(2)))
        %                     continue;
        %                 end
        %                 j1 = i1 + P.offset(1);
        %                 j2 = i2 + P.offset(2);
        %             end
        %         end

    end

end

% Domain boundaries
a = rectangle('Position',[0.0 0.0 grid.domainSize(1:2)]);
set(a,'linewidth',5);
set(a,'edgecolor','black');
xlabel('x');
ylabel('y');
title(sprintf('Grid, numLevels = %d, # total vars = %d',grid.numLevels,grid.totalVars));
axis equal;
eval(sprintf('print -depsc %s',fileName));
