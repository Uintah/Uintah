function k=plotShapeFunctions

%____________________________________________________________________
% This function plots the shape functions
% It uses functions that are defined in 
%     initializationFunctions.m
%     shapeFunctions.m
%______________________________________________________________________
  close all;
  clear all;
  global PPC;
  global NSFN;
  global d_doPlotShapeFunction;
  %______________________________________________________________________
  
  PPC = 2;                                % particles per cell
  maxpts        = int32(100);             % number of points in plot
  interpolation = 'GIMP';               %  'linear' or 'GIMP'
  domain        = 1;                      % domain length 
  dx_L          = 1;                      % dx in Left region
  lp            = dx_L/(2 * PPC)          % particle 1/2 width
  d_smallNum    = double(1e-16);
  
  %______________________________________________________________________
  
  d_doPlotShapeFunction = true;
  [sf]  = shapeFunctions;                 % load all the shape functions
  [IF]  = initializationFunctions         % load initialization functions
  
  [Regions, nRegions,NN] = IF.initialize_Regions(domain,PPC,dx_L,interpolation, d_smallNum);
  
  [nodePos]  = IF.initialize_NodePos(NN, Regions, nRegions, interpolation)
  [Lx]       = IF.initialize_Lx(NN, nodePos)

  %__________________________________
  %             GIMP  dx_L == dx_R
  %Pos:    -2     dx   -1            0            1            2
  %        |-----xxxxxxx|xxxxxxxxxxxx|xxxxxxxxxxxx|xxxxxx------|
  %Node:  (1)           2            3            4            5
  %            -L-lp                 0                  L+lp
  %
  %
  %             GIMP  dx_L= 1,  dx_R = 0.5
  %Pos:    -2     dx   -1            0      0.5     1
  %        |-----xxxxxxx|xxxxxxxxxxxx|xxxxxx|xxx----|
  %Node:  (1)           2            3      4       5
  %            (-dx-lp)_L                 0             (dx+lp)_R
  %               
  % loop over all positions from (-L -lp)_L to (L + lp)_R relative to focus node (3)  
  %
  %__________________________________
  %             LINEAR  dx_L == dx_R
  %Pos:   -1            0            1
  %        |xxxxxxxxxxxx|xxxxxxxxxxxx|
  %Node:  (1)           2            3
  %       -L            0            L
  %
  %             LINEAR  dx_L =1, dx_R = 0.5
  %Pos:   -1            0     0.5
  %        |xxxxxxxxxxxx|xxxxx|
  %Node:  (1)           2     3
  %               
  % loop over all positions from (-1) to (0.5) relative to focus node (2)
  %__________________________________
  R_L = Regions{1};                
  R_R = Regions{nRegions};         
   
  if( strcmp(interpolation,'GIMP') )
    NSFN      = 3;                        % Number of shape function nodes Linear:2, GIMP:3
    dx_L      = R_L.dx;  
    dx_R      = R_R.dx; 
    
    xp(1)     =  -dx_L - lp ;             % starting postition of particle GIMP
    xp_min    = xp(1);
    xp_max    = dx_R + dx_R/2;            % ending position
    delX      = ( xp_max - xp_min )/double(maxpts)    
    focusNode = 3;
    doBulletProofing = false;
    
  else
    NSFN      = 2;                        % Number of shape function nodes Linear:2, GIMP:3
    xp(1)     = R_L.min;
    xp_min    = xp(1);
    xp_max    = R_R.max;
    delX      = ( xp_max - xp_min )/double(maxpts)
    focusNode = 2;
    doBulletProofing = true;
  end

  %__________________________________
  % pre-allocate variables for speed
  Ss1   = zeros(maxpts,1); 
  Ss2   = zeros(maxpts,1); 
  Gs1   = zeros(maxpts,1); 
  Gs2   = zeros(maxpts,1); 
  SumS1 = zeros(maxpts,1); 
  SumS2 = zeros(maxpts,1);
  SumGs1 = zeros(maxpts,1);
  SumGs2 = zeros(maxpts,1);

  for (c=2:maxpts)
    xp(c) = xp(c-1) + delX;
    
    if( strcmp(interpolation,'GIMP') )
      [nodes, S1, SumS1(c) ]     = sf.findNodesAndWeights_gimp( xp(c), lp, nRegions, Regions, nodePos, Lx, doBulletProofing);
      [nodes, S2, SumS2(c) ]     = sf.findNodesAndWeights_gimp2(xp(c), lp, nRegions, Regions, nodePos, Lx, doBulletProofing);
    
      [nodes, G1, dx, SumG1(c) ] = sf.findNodesAndWeightGradients_gimp( xp(c), lp, nRegions, Regions, nodePos,Lx, doBulletProofing);
      [nodes, G2, dx, SumG2(c) ] = sf.findNodesAndWeightGradients_gimp2(xp(c), lp, nRegions, Regions, nodePos,Lx, doBulletProofing);
    else
      [nodes, S1, SumS1(c) ]      = sf.findNodesAndWeights_linear( xp(c), lp, nRegions, Regions, nodePos, Lx, doBulletProofing);
      [nodes, G1, dx, SumG1(c) ]  = sf.findNodesAndWeightGradients_linear( xp(c), lp, nRegions, Regions, nodePos,Lx);
      S2 = S1;   % so you don't have to put conditional statements everywhere
      G2 = G1;
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
  
  if( strcmp(interpolation,'GIMP') )
    %__________________________________
    % plot shape Function and gradient of shape function
    set(gcf,'position',[50,100,900,900]);
    subplot(2,1,1),plot(xp, Ss1, xp, Ss2)
    tmp = sprintf('Shape Function, PPC %g: dx_R = dx_L/2',PPC);
    title(tmp);
    xlabel('xp')
    legend('Single Level','multi-level');

    subplot(2,1,2),plot(xp, Gs1, 'b+', xp, Gs2, 'r.', xp,ML_grad, xp,ML_grad2)
    title('Gradient of the Shape Function');
    xlabel('xp')
    legend('Single Level','multi-level', 'Numerically Differentiated', 'Numerically Differentiated');


    %__________________________________
    % plot 1.0 - Sum (shape Function) and sum (gradient of shape function)

    figure(2)
    set(gcf,'position',[950,100,900,900]);
    diff1 = 1.0 - SumS1;
    diff2 = 1.0 - SumS2;
    
    subplot(2,1,1),plot(xp,diff1, xp, diff2)
    tmp = sprintf('1.0 - (Sum Shape Function), PPC %g: dx_R = dx_L/2',PPC);
    title(tmp);
    xlabel('xp')
    ylim([-0.1 0.1])
    legend('Single Level','multi-level');

    subplot(2,1,2),plot(xp,SumGs1, 'b+', xp,SumGs2, 'r.')
    title('Sum (Gradient of the Shape Function)');
    xlabel('xp')
    ylim([-0.1 0.1])
    legend('Single Level','multi-level'); 
    
  else
    %__________________________________
    % plot shape Function and gradient of shape function
    set(gcf,'position',[50,100,900,900]);
    subplot(2,1,1)
    plot( xp, Ss1 )
    tmp = sprintf('Linear Shape Function, PPC %g: dx_R = dx_L/2',PPC);
    title(tmp);
    xlabel('xp')
    legend('multi-Level');

    subplot(2,1,2)
    plot( xp, Gs1, 'b+', xp, ML_grad )
    title('Gradient of the Shape Function');
    xlabel('xp')
    legend('multi-Level', 'Numerically Differentiated');


    %__________________________________
    % plot Sum (shape Function) and sum (gradient of shape function)

    figure(2)
    set(gcf,'position',[950,100,900,900]);
    diff1 = 1.0 - SumS1;
    subplot(2,1,1),plot(xp,diff1)
    tmp = sprintf('1.0 - Sum(Shape Function), PPC %g: dx_R = dx_L/2',PPC);
    title(tmp);
    ylim([-0.1 0.1])
    xlabel('xp')
    legend('multi-level');

    subplot(2,1,2),plot(xp,SumGs1, 'b+')
    title('Sum (Gradient of the Shape Function)');
    xlabel('xp')
    legend('multi-level');  
  end 
  
end
