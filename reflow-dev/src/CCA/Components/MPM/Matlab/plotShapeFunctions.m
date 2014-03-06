function k=plotShapeFunctions

%____________________________________________________________________
% This function plots the shape functions
  close all;
  clear all;
  global PPC;
  global NSFN;
  PPC = 8;
 [sf]  = shapeFunctions;                 % load all the shape functions
 [IF]  = initializationFunctions   % load initialization functions

  interpolation = 'GIMP';
  domain        = 1;
  dx            = 1;
  d_smallNum    = double(1e-16);
  
  [Regions, nRegions,NN] = IF.initialize_Regions(domain,PPC,dx,interpolation, d_smallNum);
  R = Regions{1};
  
  [nodePos]  = IF.initialize_NodePos(NN, dx, Regions, nRegions, interpolation)
  
  [Lx]       = IF.initialize_Lx(NN, nodePos);
  
  Lx(2,1) = dx/4;
%  Lx(5,1) = dx/4;
  Lx


  L  = dx;
  lp = R.lp

  maxpts = int32(100);
  if( strcmp(interpolation,'GIMP') )
    NSFN      = 3;                        % Number of shape function nodes Linear:2, linear:3
    xp(1)     =  -L-lp ;                  % starting postition GIMP
    delX      = ( 2*(L + lp))/double(maxpts-1)
    focusNode = 3;
    
  else
    NSFN      = 2;                        % Number of shape function nodes Linear:2, linear:3
    xp(1)     = -L;
    delX      = 2*L/double(maxpts)
    focusNode = 2;
  end
  
  
  %             GIMP
  %Pos:    -2     dx   -1            0            1            2
  %        |-----xxxxxxx|xxxxxxxxxxxx|xxxxxxxxxxxx|xxxxxx------|
  %Node:  (1)           2            3            4            5
  %            -L-lp                 0                  L+lp
  %               
  % loop over all cells from (-L -lp) to (L + lp) relative to focus node (3)
  
  
    %             LINEAR
  %Pos:   -1            0            1
  %        |xxxxxxxxxxxx|xxxxxxxxxxxx|
  %Node:  (1)           2            3
  %       -L            0            L
  %               
  % loop over all cells from (-L) to (L) relative to focus node (2)
  
  
  Ss1(1) = 0.0;
  Ss2(1) = 0.0;
  
  Gs(1)  = 0.0;
  Gs2(1) = 0.0;

  for (c=2:maxpts)
    xp(c) = xp(c-1) + delX;
    
    if( strcmp(interpolation,'GIMP') )
      [nodes,S1]    = sf.findNodesAndWeights_gimp( xp(c), lp, nRegions, Regions, nodePos, Lx);
      [nodes,S2]    = sf.findNodesAndWeights_gimp2(xp(c), lp, nRegions, Regions, nodePos, Lx);
    
      [nodes,G1, dx]= sf.findNodesAndWeightGradients_gimp( xp(c), lp, nRegions, Regions, nodePos,Lx);
      [nodes,G2, dx]= sf.findNodesAndWeightGradients_gimp2(xp(c), lp, nRegions, Regions, nodePos,Lx);
    else
      [nodes,S1]    = sf.findNodesAndWeights_linear( xp(c), lp, nRegions, Regions, nodePos, Lx);
      [nodes,S2]    = sf.findNodesAndWeights_linear(xp(c), lp, nRegions, Regions, nodePos, Lx);
    
      [nodes,G1, dx]= sf.findNodesAndWeightGradients_linear( xp(c), lp, nRegions, Regions, nodePos,Lx);
      [nodes,G2, dx]= sf.findNodesAndWeightGradients_linear(xp(c), lp, nRegions, Regions, nodePos,Lx);
    end
    % find the index that corresponds to the focusNode
    for index=1:length(nodes)
      if(nodes(index) == focusNode)
        break
      end
    end

    Ss1(c) = S1(index);
    Ss2(c) = S2(index);
    
    Gs1(c) = G1(index);
    Gs2(c) = G2(index);
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
  tmp = sprintf('Shape Function, PPC %g: dx_R = dx_L/4',PPC);
  title(tmp);
  xlabel('xp')
  legend('Single Level','multi-level');
  
  subplot(2,1,2),plot(xp,Gs1, 'b+', xp,Gs2, 'r.', xp,ML_grad,xp,ML_grad2)
  title('Gradient of the Shape Function');
  xlabel('xp')
  legend('Single Level','multi-level', 'Numerically Differentiated', 'Numerically Differentiated');
end
