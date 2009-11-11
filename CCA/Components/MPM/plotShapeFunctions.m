function k=plotShapeFunctions

%____________________________________________________________________
% This function plots the shape functions
  close all;
  clear all;
  global PPC;
  global NSFN;
  PPC = 2;
  NSFN = 1;

  numRegions    = int32(1);              % partition the domain into numRegions
  Regions       = cell(numRegions,1);    % array that holds the individual region information

  R.min         = 0;                     % location of left node
  R.max         = 1;                     % location of right node
  R.dx          = 1;
  R.NN          = 2;                    % number of nodes including the BC nodes
  Regions{1}    = R;
  
  Lx      = zeros(R.NN,2);
  nodePos = zeros(R.NN,1);      % node Position
  nodePos(1)    = R.min;  
  nodePos(2)    = R.max;

  Lx(1,1) = R.dx;
  Lx(1,2) = R.dx/4;
  Lx(2,1) = R.dx;
  Lx(2,2) = R.dx;
  
  L = R.dx;
  lp = R.dx/(2 * PPC)

  maxpts = int32(100);
  xp(1)  = -L-lp + R.min;
  
  Ss1(1) = 0.0;
  Ss2(1) = 0.0;
  
  Gs(1)  = 0.0;
  Gs2(1) = 0.0;
  
  delX = ( 2 * (L+lp))/double(maxpts);

  for (c=2:maxpts)
    xp(c) = xp(c-1) + delX;
    [nodes,S1]    =findNodesAndWeights_gimp(xp(c), numRegions, Regions, nodePos, Lx);
    [nodes,S2]    =findNodesAndWeights_gimp2(xp(c), numRegions, Regions, nodePos, Lx);
    
    [nodes,G1, dx]=findNodesAndWeightGradients_gimp(xp(c), numRegions, Regions, nodePos);
    [nodes,G2, dx]=findNodesAndWeightGradients_gimp2(xp(c), numRegions, Regions, nodePos,Lx);    
    
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
%______________________________________________________________________
% functions
%______________________________________________________________________
%
function[nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos)
  
  nodes(1) = 1;              %HARDWIRED for this test problem
  dx = Regions{1}.dx;
end

%__________________________________
function [nodes,Ss]=findNodesAndWeights_gimp(xp, nRegions, Regions, nodePos, Lx)
  global PPC;
  global NSFN;
  
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.  This follows the numenclature
  % of equation 7.16 of the MPM documentation 

 [nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos);
  
  L = dx;
  lp= dx/(2 * PPC);          % This assumes that lp = lp_initial.
  
  for ig=1:NSFN
    Ss(ig) = double(-9);
    delX = xp - nodePos(nodes(ig));

    if ( ((-L-lp) < delX) && (delX <= (-L+lp)) )
      
      Ss(ig) = ( ( L + lp + delX)^2 )/ (4.0*L*lp);
      
    elseif( ((-L+lp) < delX) && (delX <= -lp) )
      
      Ss(ig) = 1 + delX/L;
      
    elseif( (-lp < delX) && (delX <= lp) )
      
      numerator = delX^2 + lp^2;
      Ss(ig) =1.0 - (numerator/(2.0*L*lp));  
    
    elseif( (lp < delX) && (delX <= (L-lp)) )
      
      Ss(ig) = 1 - delX/L;
            
    elseif( ((L-lp) < delX) && (delX <= (L+lp)) )
    
      Ss(ig) = ( ( L + lp - delX)^2 )/ (4.0*L*lp);
    
    else
      Ss(ig) = 0;
    end
  end
  
end

%__________________________________
function [nodes,Ss]=findNodesAndWeights_gimp2(xp, nRegions, Regions, nodePos, Lx)
  global PPC;
  global NSFN;
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.  This follows the numenclature
  % of equation 15 of the reference
  [nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos);
 
  for ig=1:NSFN
    node = nodes(ig);
    Lx_minus = Lx(node,1);
    Lx_plus  = Lx(node,2);
    lp       = dx/(2 * PPC);          % This assumes that lp = lp_initial.

    delX = xp - nodePos(node);
    A = delX - lp;
    B = delX + lp;
    a = max( A, -Lx_minus);
    b = min( B,  Lx_plus);

    if (B <= -Lx_minus || A >= Lx_plus)
      
      Ss(ig) = 0;
    
    elseif( b <= 0 )
    
      t1 = b - a;
      t2 = (b*b - a*a)/(2.0*Lx_minus);
      Ss(ig) = (t1 + t2)/(2.0*lp);
      
    elseif( a >= 0 )
      
      t1 = b - a;
      t2 = (b*b - a*a)/(2.0*Lx_plus);
      Ss(ig) = (t1 - t2)/(2.0*lp);
    else
    
      t1 = b - a;
      t2 = (a*a)/(2.0*Lx_minus);
      t3 = (b*b)/(2.0*Lx_plus);
      Ss(ig) = (t1 - t2 - t3)/(2*lp);
      
    end
  end
  
end

%__________________________________
function [nodes,Gs, dx]=findNodesAndWeightGradients_gimp(xp, nRegions, Regions, nodePos)
  global PPC;
  global NSFN;
  % find the nodes that surround the given location and
  % the values of the gradients of the shape functions.
  % Assume the grid starts at x=0.
  [nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos);
  
  L  = dx;
  lp = dx/(2 * PPC);          % This assumes that lp = lp_initial.
  
  for ig=1:NSFN
    Gs(ig) = -9;
    delX = xp - nodePos(nodes(ig));

    if ( ((-L-lp) < delX) && (delX <= (-L+lp)) )
      
      Gs(ig) = ( L + lp + delX )/ (2.0*L*lp);
      
    elseif( ((-L+lp) < delX) && (delX <= -lp) )
      
      Gs(ig) = 1/L;
      
    elseif( (-lp < delX) && (delX <= lp) )
      
      Gs(ig) =-delX/(L*lp);  
    
    elseif( (lp < delX) && (delX <= (L-lp)) )
      
      Gs(ig) = -1/L;
            
    elseif( ( (L-lp) < delX) && (delX <= (L+lp)) )
    
      Gs(ig) = -( L + lp - delX )/ (2.0*L*lp);
    
    else
      Gs(ig) = 0;
    end
  end
end

%__________________________________
function [nodes,Gs, dx]=findNodesAndWeightGradients_gimp2(xp, nRegions, Regions, nodePos, Lx)
  global PPC;
  global NSFN;
  % find the nodes that surround the given location and
  % the values of the gradients of the shape functions.
  % Assume the grid starts at x=0.
  [nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos);
  
  ig = 1;
  
  node = nodes(ig);                                                       
  Lx_minus = Lx(node,1);                                                  
  Lx_plus  = Lx(node,2);                                                  
  lp       = dx/(2 * PPC);          % This assumes that lp = lp_initial.  

  delX = xp - nodePos(node);                                              
  A = delX - lp;                                                          
  B = delX + lp;                                                          
  a = max( A, -Lx_minus);                                                 
  b = min( B,  Lx_plus);                                                  
                                                           
  if (B <= -Lx_minus || A >= Lx_plus)   %--------------------------     B<= -Lx- & A >= Lx+                                    

    Gs(ig) = 0;                                                           

  elseif( b <= 0 )                      %--------------------------     b <= 0                                                

    if( (B < Lx_plus) && (A > -Lx_minus))

      Gs(ig) = 1/Lx_minus;

    elseif( (B >= Lx_plus) && (A > -Lx_minus) )

      Gs(ig) = (Lx_minus + A)/ (2.0 * Lx_minus * lp);

    elseif( (B < Lx_plus) && (A <= -Lx_minus) )

      Gs(ig) = (Lx_minus + B)/ (2.0 * Lx_minus * lp);      

    else
      Gs(ig) = 0.0;
    end                            

  elseif( a >= 0 )                        %--------------------------    a >= 0                                          

    if( (B < Lx_plus) && (A > -Lx_minus))

      Gs(ig) = -1/Lx_plus;

    elseif( (B >= Lx_plus) && (A > -Lx_minus) )

      Gs(ig) = (-Lx_plus + A)/ (2.0 * Lx_plus * lp);

    elseif( (B < Lx_plus) && (A <= -Lx_minus) )

      Gs(ig) = (Lx_plus - B)/ (2.0 * Lx_plus * lp);      

    else
      Gs(ig) = 0.0;
    end  

  else                                      %--------------------------    other                                                    

     if( (B < Lx_plus) && (A > -Lx_minus))

      Gs(ig) = -A/(2.0 * Lx_minus * lp)  - B/(2.0 * Lx_plus * lp);

    elseif( (B >= Lx_plus) && (A > -Lx_minus) )

      Gs(ig) = (-Lx_minus - A)/(2.0 * Lx_minus * lp)

    elseif( (B < Lx_plus) && (A <= -Lx_minus) )

      Gs(ig) = (Lx_plus - B)/(2.0 * Lx_plus * lp)

    else
      Gs(ig) = 0.0;
    end                                          
  end
end
