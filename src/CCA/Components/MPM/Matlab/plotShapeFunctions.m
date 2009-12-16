function k=plotShapeFunctions

%____________________________________________________________________
% This function plots the shape functions
  close all;
  clear all;
  global PPC;
  global NSFN;
  PPC = 2;
 [sf]  = shapeFunctions;                 % load all the shape functions

  interpolation = 'GIMP';

  if( strcmp(interpolation,'GIMP') )
    NSFN    = 3;                        % Number of shape function nodes Linear:2, GIMP:3
  else
    NSFN    = 2;        
  end

  numRegions    = int32(1);              % partition the domain into numRegions
  Regions       = cell(numRegions,1);    % array that holds the individual region information

  R.min         = 0;                     % location of left node
  R.max         = 1;                     % location of right node
  R.dx          = 1;
  R.NN          = 4;                    % number of nodes including the BC nodes
  Regions{1}    = R;
  
  Lx            = zeros(R.NN,2);
  nodePos       = zeros(R.NN,1);        % node Position
  nodePos(1)    = -R.dx;
  
  for  n=2:R.NN  
    nodePos(n) = nodePos(n-1) + R.dx;
  end
  
  Lx(1,1) = 0;
  Lx(1,2) = R.dx;
  Lx(2,1) = R.dx;
  Lx(2,2) = R.dx;
  Lx(3,1) = R.dx;
  Lx(3,2) = R.dx;
  Lx(4,1) = R.dx;
  Lx(4,2) = 0;
  
  L = R.dx;
  lp = R.dx/(2 * PPC);

  maxpts = int32(100);
  xp(1)  =  R.min;
  
  Ss1(1) = 0.0;
  Ss2(1) = 0.0;
  
  Gs(1)  = 0.0;
  Gs2(1) = 0.0;
  
  delX = ( 2 * (L+lp))/double(maxpts);

  for (c=2:maxpts)
    xp(c) = xp(c-1) + delX;
    [nodes,S1]    = sf.findNodesAndWeights_gimp( xp(c), lp, numRegions, Regions, nodePos, Lx);
    [nodes,S2]    = sf.findNodesAndWeights_gimp2(xp(c), lp, numRegions, Regions, nodePos, Lx);
    
    [nodes,G1, dx]= sf.findNodesAndWeightGradients_gimp( xp(c), lp, numRegions, Regions, nodePos,Lx);
    [nodes,G2, dx]= sf.findNodesAndWeightGradients_gimp2(xp(c), lp, numRegions, Regions, nodePos,Lx);    
    
    Ss1(c) = S1(1);
    Ss2(c) = S2(1);
    
    Gs1(c) = G1(1);
    Gs2(c) = G2(1);
  end
 
  % Numerically differentiate the shape functions
  ML_grad  = diff(Ss1)/delX;
  ML_grad2 = diff(Ss2)/delX;
  ML_grad(maxpts)  = 0.0;
  ML_grad2(maxpts) = 0.0;  
 
  %__________________________________
  % plot up the results
  set(gcf,'position',[50,100,900,900]);
  subplot(2,1,1),plot(xp,Ss1, xp, Ss2)
  title('Shape Function, PPC 8: dx_R = dx_L/4');
  xlabel('xp')
  legend('Single Level','multi-level');
  
  subplot(2,1,2),plot(xp,Gs1, 'b+', xp,Gs2, 'r.', xp,ML_grad,xp,ML_grad2)
  title('Gradient of the Shape Function');
  xlabel('xp')
  legend('Single Level','multi-level', 'Numerically Differentiated', 'Numerically Differentiated');
end
