function plotGrid(grid,fileName,showGhosts,showCellIndex,showCircles,showUnused)
%PLOTGRID  Plot 2D grid.
%   PLOT(GRID,FILENAME,SHOWGHOSTS,SHOWCELLINDEX,SHOWCIRCLES,SHOWUNUSED)
%   plots a 2D (works only in 2D forn now) grid hierarchy GRID, and prints it
%   to the filename FILENAME in eps format (default FILENAME = 'grid.eps').
%   options:
%   'SHOWGHOSTS'        Show ghost cells.
%   'SHOWCELLINDEX'     Show cell global 1D index of each cell.
%   'SHOWCIRCLES'       Plot circles at cell centers to indicate where the
%                       data is.
%   'SHOWUNUSED'        Plot cells that are not used in the actual discretization.
%
%   See also: PLOTRESULTS, TESTDISC, TESTADATPIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

if (nargin < 2)
    fileName = 'grid.eps';
end
if (nargin < 3)
    showGhosts = 0;
end
if (nargin < 4)
    showCellIndex = 0;
end
if (nargin < 5)
    showCellIndex = 0;
end
if (nargin < 6)
    showUnused = 0;
end

switch (grid.dim)
    case 1,

        figure(1);
        clf;

        levelColor = {'blue','red','green','black','cyan','blue','red','green','black','cyan',...
            'blue','red','green','black','cyan','blue','red','green','black','cyan'};
        levelWidth = [3 3 3 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];

        for k = 1:grid.numLevels,
            level = grid.level{k};
            h = level.h;
            hy = 0.5;
            if (k == 1)
                offsetLower = [-1.5 -1.5];
                offsetUpper = [0.5 0.5];
                if (showGhosts)
                    offsetLower = offsetLower - 1;
                    offsetUpper = offsetUpper + 1;
                end
                axis([...
                    (level.minCell(1)+offsetLower(1))*h(1) (level.maxCell(1)+offsetUpper(1))*h(1) ...
                    -hy hy
                    ]);
                hold on;
            end

            for q = 1:grid.level{k}.numPatches,
                P = grid.level{k}.patch{q};

                i1          = [P.ilower(1)-1:P.iupper(1)+1];
                mat1        = i1;
                x1          = (mat1-0.5)*h(1);
                x1          = x1(:);

                for i = 1:length(x1)
                    index = P.cellIndex(mat1(i)+P.offsetSub(1));
                    if ((~showUnused) & (ismember(index,level.indUnused)))
                        continue;
                    end

                    if ((mat1(i) >= P.ilower(1)) & (mat1(i) <= P.iupper(1)))
                        a = rectangle('Position',[x1(i)-0.5*h(1) -0.5*hy h(1) hy]);
                        set(a,'edgecolor',levelColor{k});
                        set(a,'linewidth',levelWidth(k));
                        set(a,'linestyle','-')

                        if (showCircles)
                            b = plot(x1(i),0.0,'o');
                            set(b,'markersize',10);
                            set(b,'markeredgecolor',levelColor{k});
                            %set(b,'markerfacecolor',levelColor{k});
                            set(b,'markerfacecolor','white');
                        end

                        if (showCellIndex)
                            c = text(x1(i)-0.125*h(1),-0.125*hy,sprintf('%d',index));
                            set(c,'fontsize',10);
                            set(c,'color','k');
                        end

                    elseif (showGhosts)
                        a = rectangle('Position',[x1(i)-0.5*h(1) -0.5*hy h(1) hy]);
                        set(a,'edgecolor',levelColor{k});
                        set(a,'linewidth',levelWidth(k));
                        set(a,'linestyle','--')

                        if (showCircles)
                            b = plot(x1(i),0.0,'^');
                            set(b,'markersize',10);
                            set(b,'markeredgecolor',levelColor{k});
                            set(b,'markerfacecolor','white');
                        end

                        if (showCellIndex)
                            index = P.cellIndex(mat1(i)+P.offsetSub(1));
                            c = text(x1(i)-0.125*h(1),-0.125*hy,sprintf('%d',index));
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
                %                 j1 = i1 + P.offsetSub(1);
                %                 j2 = i2 + P.offsetSub(2);
                %             end
                %         end

            end

        end

        % Domain boundaries
        a = rectangle('Position',[-0.1 -0.6*hy 1.2*grid.domainSize(1) 1.2*hy]);
        set(a,'linewidth',5);
        set(a,'edgecolor','black');

        xlabel('x');
        title(sprintf('Grid, numLevels = %d, # total vars = %d',grid.numLevels,grid.totalVars));
        %        axis;
        eval(sprintf('print -depsc %s',fileName));

    case 2,

        figure(1);
        clf;

        levelColor = {'blue','red','green','black','cyan','blue','red','green','black','cyan',...
            'blue','red','green','black','cyan','blue','red','green','black','cyan'};
        levelWidth = [3 3 3 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];

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
                    index = P.cellIndex(mat1(i)+P.offsetSub(1),mat2(i)+P.offsetSub(2));
                    if ((~showUnused) & (ismember(index,level.indUnused)))
                        continue;
                    end

                    if (    (mat1(i) >= P.ilower(1)) & (mat1(i) <= P.iupper(1)) & ...
                            (mat2(i) >= P.ilower(2)) & (mat2(i) <= P.iupper(2)))
                        a = rectangle('Position',[x1(i)-0.5*h(1) x2(i)-0.5*h(2) h(1) h(2)]);
                        set(a,'edgecolor',levelColor{k});
                        set(a,'linewidth',levelWidth(k));
                        set(a,'linestyle','-')

                        if (showCircles)
                            b = plot(x1(i),x2(i),'o');
                            set(b,'markersize',10);
                            set(b,'markeredgecolor',levelColor{k});
                            %set(b,'markerfacecolor',levelColor{k});
                            set(b,'markerfacecolor','white');
                        end

                        if (showCellIndex)
                            c = text(x1(i)-0.125*h(1),x2(i)-0.125*h(2),sprintf('%d',index));
                            set(c,'fontsize',10);
                            set(c,'color','k');
                        end

                    elseif (showGhosts)
                        a = rectangle('Position',[x1(i)-0.5*h(1) x2(i)-0.5*h(2) h(1) h(2)]);
                        set(a,'edgecolor',levelColor{k});
                        set(a,'linewidth',levelWidth(k));
                        set(a,'linestyle','--')

                        if (showCircles)
                            b = plot(x1(i),x2(i),'^');
                            set(b,'markersize',10);
                            set(b,'markeredgecolor',levelColor{k});
                            set(b,'markerfacecolor','white');
                        end

                        if (showCellIndex)
                            index = P.cellIndex(mat1(i)+P.offsetSub(1),mat2(i)+P.offsetSub(2));
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
                %                 j1 = i1 + P.offsetSub(1);
                %                 j2 = i2 + P.offsetSub(2);
                %             end
                %         end

            end

        end

        % Domain boundaries
        switch (param.problemType)

            case 'Lshaped',     % Delete upper-rigt quadrant
                a = line([0.0 grid.domainSize(1)],[0.0 0.0]);
                set(a,'linewidth',5);
                set(a,'color','black');

                a = line([grid.domainSize(1) grid.domainSize(1)],[0 0.5*grid.domainSize(2)]);
                set(a,'linewidth',5);
                set(a,'color','black');

                a = line([grid.domainSize(1) 0.5*grid.domainSize(1)],[0.5*grid.domainSize(2) 0.5*grid.domainSize(2)]);
                set(a,'linewidth',5);
                set(a,'color','black');

                a = line([0.5*grid.domainSize(1) 0.5*grid.domainSize(1)],[0.5*grid.domainSize(2) grid.domainSize(2)]);
                set(a,'linewidth',5);
                set(a,'color','black');

                a = line([0.5*grid.domainSize(1) 0.0],[grid.domainSize(2) grid.domainSize(2)]);
                set(a,'linewidth',5);
                set(a,'color','black');

                a = line([0.0 0.0],[grid.domainSize(2) 0.0]);
                set(a,'linewidth',5);
                set(a,'color','black');

            otherwise,
                a = rectangle('Position',[0.0 0.0 grid.domainSize(1:2)]);
                set(a,'linewidth',5);
                set(a,'edgecolor','black');
        end

        xlabel('x');
        ylabel('y');
        title(sprintf('Grid, numLevels = %d, # total vars = %d',grid.numLevels,grid.totalVars));
        axis equal;
        eval(sprintf('print -depsc %s',fileName));

    otherwise,
        fprintf('Warning: plotGrid not supported yet for numDims = %d\n',grid.dim);
end
