%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Program:      DQMOM Driver File
%
% Author:       Charles Reid
%
% Description:  Driver file that solves DQMOM weight/weighted abscissa transport equations and advances
%               them in time by calling DQMOM AX=B solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear all;
clc;

W = 1.0; L = 1.0;               % geometry
nx = 4; ny = 4;                 % resolution
%nx = nx + 2; ny = ny + 2;       % add ghost cells (ignore for now)
%dx = L/(nx-2); dy = W/(ny-2);   % grid spacing
dx = L/nx; dy = W/ny;
vol = dx*dy;                    % cell volume

%[x,y]=meshgrid(linspace(-dx/2,L+dx/2,nx),linspace(-dy/2,W+dx/2,ny));
x=linspace(0,L,nx);
y=linspace(0,W,ny);

% assemble w, wa, G, k
wa = ([5;
       3;
       1]);

w = [0.5];

k = ([1,2,1;
      2,1,3;
      3,2,0;
      0,1,1]);

% k1 = ([1,2;
%        2,1;
%        0,1]);
% 
% k2 = ([0,1;
%        2,0;
%        1,1]);

N_xi = size(wa,1);
N = size(w,2);
Ntot = (N_xi+1)*N;

% initialize weighted abscissas to these values
for i=1:nx
    for j=1:ny/2
        for alpha=1:N
            % initialize weights array
            w_all(i,j,alpha) = w(alpha);
%             w_all1(i,j,alpha) = w(alpha);
%             w_all2(i,j,alpha) = w(alpha);
            
            % initialize weighted abscissas array
            for m=1:N_xi
                wa_all(i,j,m,alpha) = wa(m,alpha);
%                 wa_all1(i,j,m,alpha) = wa(m,alpha);
%                 wa_all2(i,j,m,alpha) = wa(m,alpha);
            end
        end
    end

    for j=(ny/2+1):ny
        for alpha=1:N
            % initialize weights array
            w_all(i,j,alpha) = w(alpha);
%             w_all1(i,j,alpha) = w(alpha);
%             w_all2(i,j,alpha) = w(alpha);
            
            %initialize weighted abscissas to be zero
            for m=1:N_xi
                % this can't be zero
                % otherwise some of the moments have ^-1 power
                % so this is 1/0
                % (even if you have large moments, this can still be INF -
                % not sure why)
                wa_all(i,j,m,alpha) = 0.00001;
%                 wa_all1(i,j,m,alpha) = 0.00001;
%                 wa_all2(i,j,m,alpha) = 0.00001;
            end
        end
    end
    
end

for i=1:(nx/2)
    for j=1:(ny/2)
        for alpha=1:N
            % initialize weights array
            w_all(i,j,alpha) = w(alpha);
%             w_all1(i,j,alpha) = w(alpha);
%             w_all1(i,j,alpha) = w(alpha);
            
            for m=1:N_xi
                % lower left hand corner in surf plots
                wa_all(i,j,m,alpha) = 2;
%                 wa_all1(i,j,m,alpha) = 2;
%                 wa_all2(i,j,m,alpha) = 2;
            end
        end
    end
end

%---------------------------------------------
% Start the time-stepping

count = 1;
t=1;
ttime = 10; %s
delta_t = 0.1; %s
while t<=ttime
  
    fprintf('------------------------------------------------------\n');
    fprintf('Time step = %0.2f\n',t);

    for i=1:nx
        for j=1:ny
            
            %fprintf('Entering cell i=%0.0f, j=%0.0f\n',i,j);
            % grab weights and weighted abscissas at current location
            for alpha=1:N
                w_temp(alpha) = w_all(i,j,alpha);
%                 w_temp1(alpha) = w_all1(i,j,alpha);
%                 w_temp2(alpha) = w_all2(i,j,alpha);
                for m=1:N_xi
                    wa_temp(m,alpha)=wa_all(i,j,m,alpha);
%                     wa_temp1(m,alpha)=wa_all1(i,j,m,alpha);
%                     wa_temp2(m,alpha)=wa_all2(i,j,m,alpha);
                end
            end
            
            % here you can specify a "growth" term
%             for m=1:N_xi
%                 for alpha=1:N
%                     G(m,alpha) = (wa_temp(m,alpha)/w_temp(alpha))*sin(t);
% %                     G1(m,alpha) = (wa_temp1(m,alpha)/w_temp1(alpha))*sin(t);
% %                     G2(m,alpha) = (wa_temp2(m,alpha)/w_temp2(alpha))*sin(t);
%                 end
%             end
            
            G=([15;
                10;
                0]);
            
            % solve DQMOM linear system
            X = dqmom_linear_system(w_temp, wa_temp, k, G);
%             X1 = dqmom_linear_system(w_temp1, wa_temp1, k1, G1);
%             X2 = dqmom_linear_system(w_temp2, wa_temp2, k2, G2);

            % update weight/weighted abscissa values
            for m=0:N_xi
                for alpha=1:N
                    if (m==0)
                        % update weights
                        %fprintf('Old value of weight alpha=%0.0f is %0.4f\n',alpha,w_all(i,j,alpha));
                        
                        w_all(i,j,alpha) = w_all(i,j,alpha) + delta_t*(X(alpha));
%                         w_all1(i,j,alpha) = w_all1(i,j,alpha) + delta_t*(X1(alpha));
%                         w_all2(i,j,alpha) = w_all2(i,j,alpha) + delta_t*(X2(alpha));
                        
                        %fprintf('New value of weight alpha=%0.0f is %0.4f\n',alpha,w_all(i,j,alpha));
                        %fprintf('Weight at quad node %0.0f is gaining value %0.4f\n',alpha,delta_t*(X(alpha)));
                    else
                        % update weighted abscissas
                        %fprintf('Old value of weighted abscissa %0.0f, alpha=%0.0f is %0.4f\n',m,alpha,wa_all(i,j,m,alpha));
                        
                        wa_all(i,j,m,alpha) = wa_all(i,j,m,alpha) + delta_t*(X(m*N + alpha));
%                         wa_all1(i,j,m,alpha) = wa_all1(i,j,m,alpha) + delta_t*(X1(m*N + alpha));
%                         wa_all2(i,j,m,alpha) = wa_all2(i,j,m,alpha) + delta_t*(X2(m*N + alpha));
                        
                        %fprintf('New value of weighted abscissa %0.0f, alpha=%0.0f is %0.4f\n',m,alpha,wa_all(i,j,m,alpha));
                        %fprintf('Weighted abscissa for internal coordinate %0.0f at quad node %0.0f is gaining value %0.4f\n',j,alpha,delta_t*(X(m*N+alpha)));
                    end
                end
            end
            
        end
    end


    colormap(jet)
    cmin = -10;
    cmax = 20;
    zmin = -10;
    zmax = 20;
    
    % make a 2-by-1 plot of abscissas
    subplot(131)
    surf(wa_all(:,:,1,1)')
    %contourf(wa_all(:,:,1,1)')
    title('Internal Coordinate 1')
    colorbar('location','southoutside')
    caxis([cmin cmax])
    zlim([cmin cmax])
    
    subplot(132)
    surf(wa_all(:,:,2,1)')
    %contourf(wa_all(:,:,2,1)')
    title('Internal Coordinate 2')
    colorbar('location','southoutside')
    caxis([cmin cmax])
    zlim([zmin zmax])

    subplot(133)
    surf(wa_all(:,:,3,1)')
    %contourf(wa_all(:,:,2,1)')
    title('Internal Coordinate 3')
    colorbar('location','southoutside')
    caxis([cmin cmax])
    zlim([zmin zmax])

%     subplot(121)
%     surf(wa_all1(:,:,2,1)')
%     title('Abscissa 1 K 1')
%     colorbar('location','southoutside')
%     caxis([cmin cmax])
%     zlim([zmin zmax])
%     
%     subplot(122)
%     surf(wa_all2(:,:,2,1)')
%     title('Moments index 2')
%     colorbar('location','southoutside')
%     caxis([cmin cmax])
%     zlim([zmin zmax])
    
    if (t == 1)
        pause();
    else
        pause(0.02);
    end
    
    t = t + delta_t;
    count = count + 1;
    clear('X');

end
