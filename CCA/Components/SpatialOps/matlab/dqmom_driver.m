%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Program:      DQMOM Driver File
%
% Author:       Charles Reid
%
% Description:  Driver file that solves DQMOM weight/weighted abscissa transport equations and advances
%               them in time by calling DQMOM AX=B solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic

close all;
clear all;
clc;

%% Problem Geometry

W = 1.0; L = 1.0;               % geometry
nx = 2; ny = 2;               % resolution
%nx = nx + 2; ny = ny + 2;       % add ghost cells (ignore for now)
%dx = L/(nx-2); dy = W/(ny-2);   % grid spacing w/ ghost cells
dx = L/nx; dy = W/ny;           % grid spacing
vol = dx*dy;                    % cell volume

x=linspace(0,L,nx);
y=linspace(0,W,ny);

%% DQMOM Specifications

% weighted abscissa values
wa = ([1,2]);
%wa = ([1,
%       2]);

% weight values
w = [1,1];

% moment indexes
k = ([1;2;3;4]);
%k = ([0,2;
%      2,1;
%      3,0]);

% number of quad nodes, etc.
N_xi = size(wa,1);
N = size(w,2);
Ntot = (N_xi+1)*N;

% initialize weighted abscissas to these values
for i=1:nx
    %for j=1:ny/2
    for j=1:ny
        for alpha=1:N
            % initialize weights array
            w_all(i,j,alpha) = w(alpha);
            
            % initialize weighted abscissas array
            for m=1:N_xi
                wa_all(i,j,m,alpha) = wa(m,alpha);
            end
        end
    end
end

%% Simulation Controller

% -----------------------------
% Start the time-stepping

count = 1;
t=1;
ttime = 25; %s
delta_t = 0.1; %s
while t<=ttime
    
    fprintf('------------------------------------------------------\n');
    fprintf('Time step = %0.2f\n',t);

    for i=1:nx
        for j=1:ny
            
            % grab weights and weighted abscissas at current location
            for alpha=1:N
                w_temp(alpha) = w_all(i,j,alpha);
                for m=1:N_xi
                    wa_temp(m,alpha)=wa_all(i,j,m,alpha);
                end
            end
            
            % here you can specify a "growth" term
            for alpha=1:N
                G(1,alpha) = 1;
%                G(2,alpha) = 0;
            end
            
            % solve DQMOM linear system
            X = dqmom_linear_system(w_temp, wa_temp, k, G);

            % update weight/weighted abscissa values
            for m=0:N_xi
                for alpha=1:N
                    if (m==0)
                        % update weights
                        w_all(i,j,alpha) = w_all(i,j,alpha) + delta_t*(X(alpha));
                    else
                        % update weighted abscissas
                        wa_all(i,j,m,alpha) = wa_all(i,j,m,alpha) + delta_t*(X(m*N + alpha));
                    end
                end
            end
            
        end
    end
    
    % make a plot of abscissa vs time for a point in space
    %plot(t,(wa_all(2,2,1,1)/w_all(2,2,1)),'bx',t,(wa_all(2,2,1,2)/w_all(2,2,2)),'rx't,w_all(2,2,1),'ks');
    plot(t,(wa_all(2,2,1,1)/w_all(2,2,1)),'bx',t,w_all(2,2,1),'ks');
    legend('internal coord 1','weight','location','eastoutside')
    title('Wt, Int Coord 1 Abscissa at x=2,y=2 vs. time')
    hold on;

%     colormap(jet)
%     cmin = -10;
%     cmax = 20;
%     zmin = -10;
%     zmax = 20;
% 
%     % make a 3-by-1 surface plot of abscissas
%     subplot(131)
%     surf(wa_all(:,:,1,1)')
% %     contourf(wa_all(:,:,1,1)')
%     title('Internal Coordinate 1')
%     colorbar('location','southoutside')
%     caxis([cmin cmax])
%     zlim([cmin cmax])
%     
%     subplot(132)
%     surf(wa_all(:,:,2,1)')
% %     contourf(wa_all(:,:,2,1)')
%     title('Internal Coordinate 2')
%     colorbar('location','southoutside')
%     caxis([cmin cmax])
%     zlim([zmin zmax])
% 
%     subplot(133)
%     surf(w_all(:,:,1,1)')
% %     contourf(wa_all(:,:,2,1)')
%     title('Weights')
%     colorbar('location','southoutside')
%     caxis([cmin cmax])
%     zlim([zmin zmax])

    if (t == 1)
        pause();
    else
        pause(0.5);
    end
    
    t = t + delta_t;
    count = count + 1;
    clear('X');

end

toc
