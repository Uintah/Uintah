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
nx = 1; ny = 1;               % resolution
%nx = nx + 2; ny = ny + 2;       % add ghost cells (ignore for now)
%dx = L/(nx-2); dy = W/(ny-2);   % grid spacing w/ ghost cells
dx = L/nx; dy = W/ny;           % grid spacing
vol = dx*dy;                    % cell volume

x=linspace(0,L,nx);
y=linspace(0,W,ny);

%% DQMOM Specifications

% weighted abscissa values
wa = ([1, 2, 3;
       1, 2, 3]);

% weight values
w = [1, 1, 1];

% moment indexes
k = ([0,0;
      0,1;
      0,2;
      1,1;
      1,2;
      2,2;
      2,3;
      1,0;
      2,1]);

% number of quad nodes, etc.
N_xi = size(wa,1);
N = size(w,2);
Ntot = (N_xi+1)*N;

% initialize weighted abscissas to these values
for alpha=1:N
    % initialize weights array
    %w_all(i,j,alpha) = w(alpha);
    w_all(1,alpha) = w(alpha);
            
    % initialize weighted abscissas array
    for m=1:N_xi
        %wa_all(i,j,m,alpha) = wa(m,alpha);
        wa_all(1,m,alpha) = wa(m,alpha);
    end
end

%% Simulation Controller

% -----------------------------
% Start the time-stepping

count = 1;
t=1;
ttime = 50; %s
delta_t = 0.1; %s
while t<=ttime
    
    fprintf('------------------------------------------------------\n');
    fprintf('Time step = %0.2f\n',t);

%    for i=1:nx
%        for j=1:ny
            
            % grab weights and weighted abscissas at current location
            for alpha=1:N
                %w_temp(alpha) = w_all(i,j,alpha);
                w_temp(alpha) = w_all(count,alpha);
                for m=1:N_xi
                    %wa_temp(m,alpha)=wa_all(i,j,m,alpha);
                    wa_temp(m,alpha) = wa_all(count,m,alpha);
                end
            end
            
            % here you can specify a "growth" term
            for alpha=1:N
                G(1,alpha) = -1;
                G(2,alpha) = 0;
            end
            
            % solve DQMOM linear system
            [condition_number singular X] = dqmom_linear_system(w_temp, wa_temp, k, G);
            condition_numbers(count) = condition_number;
            if (singular)
              fprintf('Warning: singular A matrix!\n');
              %break;
            end

            % update weight/weighted abscissa values
            for m=0:N_xi
                for alpha=1:N
                    if (m==0)
                        % update weights
                        w_all(count+1, alpha)   = w_all(count, alpha) + delta_t*( X(alpha) );
                        w_all_src(count, alpha) = X(alpha);
                    else
                        % update weighted abscissas
                        wa_all(count+1, m, alpha) = wa_all(count, m, alpha) + delta_t*( X(m*N + alpha) );
                        wa_all_src(count, m, alpha)  = X(m*N + alpha);
                    end
                end
            end
            
%        end %for j
%    end %for i
 
    t = t + delta_t;
    count = count + 1;
    fprintf('Updating count...\n');
    clear('X');

end

subplot(4,2,1)
% 2 quad nodes:
%plot(w_all(:,1),'b-',w_all(:,2),'r-')
%legend('weight qn 0','weight qn 1','location','eastoutside')
% 3 quad nodes:
plot(w_all(:,1),'b-',w_all(:,2),'r-',w_all(:,3),'g-')
legend('weight qn 0','weight qn 1','weight qn 2','location','eastoutside')
ylabel('weights')

subplot(4,2,2)
% 2 quad nodes:
%plot(w_all_src(:,1),'b-',w_all_src(:,2),'r-')
% 3 quad nodes:
plot(w_all_src(:,1),'b-',w_all_src(:,2),'r-',w_all_src(:,3),'g-')
ylabel('weight source')

subplot(4,2,3)
% 2 quad nodes:
%plot((wa_all(:,1,1)./w(:,1)),'b-',(wa_all(:,1,2)./w(:,2)),'r-')
%legend('length qn 0','length qn1','location','eastoutside')
% 3 quad nodes:
plot((wa_all(:,1,1)./w(:,1)),'b-',(wa_all(:,1,2)./w(:,2)),'r-',(wa_all(:,1,3)./w(:,3)),'g-')
legend('length qn 0','length qn 1','length qn 2','location','eastoutside')
ylabel('length')

subplot(4,2,4)
% 2 quad nodes:
%plot((wa_all_src(:,1,1)./w(:,1)),'b-',(wa_all_src(:,1,2)./w(:,2)),'r-')
% 3 quad nodes:
plot((wa_all_src(:,1,1)./w(:,1)),'b-',(wa_all_src(:,1,2)./w(:,2)),'r-',(wa_all_src(:,1,3)./w(:,3)),'g-')
ylabel('length source')

subplot(4,2,5)
% 2 quad nodes:
%plot((wa_all(:,2,1)./w(:,1)),'b-',(wa_all(:,2,2)./w(:,2)),'r-')
%legend('mass frac qn 0','mass frac qn1','location','eastoutside')
% 3 quad nodes:
plot((wa_all(:,2,1)./w(:,1)),'b-',(wa_all(:,2,2)./w(:,2)),'r-',(wa_all(:,2,3)./w(:,3)),'g-')
legend('mass frac qn0','mass frac qn1','mass frac qn2','location','eastoutside')
ylabel('mass frac')

subplot(4,2,6)
% 2 quad nodes:
%plot((wa_all_src(:,2,1)./w(:,1)),'b-',(wa_all_src(:,2,2)./w(:,2)),'r-')
% 3 quad nodes:
plot((wa_all_src(:,2,1)./w(:,1)),'b-',(wa_all_src(:,2,2)./w(:,2)),'r-',(wa_all_src(:,2,3)./w(:,3)),'g-')
ylabel('mass frac source')

subplot(4,1,4)
semilogy(condition_numbers(:),'k-')
ylabel('condition no.')

toc
