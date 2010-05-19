function [sf] = shapeFunction()
  % create function handles that are used in AMRMPM.m
  sf.positionToNode                     = @positionToNode;
  sf.positionToClosestNodes             = @positionToClosestNodes;
  sf.positionToVolP                     = @positionToVolP;
  sf.findNodesAndWeights_linear         = @findNodesAndWeights_linear;
  sf.findNodesAndWeights_gimp           = @findNodesAndWeights_gimp;
  sf.findNodesAndWeights_gimp2          = @findNodesAndWeights_gimp2;
  sf.findNodesAndWeightGradients_linear = @findNodesAndWeightGradients_linear;
  sf.findNodesAndWeightGradients_gimp   = @findNodesAndWeightGradients_gimp;
  sf.findNodesAndWeightGradients_gimp2  = @findNodesAndWeightGradients_gimp2;
  %__________________________________
%
function[node, dx]=positionToNode(xp, nRegions, Regions)
 
  n_offset = 1;                              % the left node is owned by the right patch 
  node = int32(-9);
  
  for r=1:nRegions
    R = Regions{r};
    dx = R.dx;
    
    if ((xp >= R.min) && (xp < R.max))   
      n    = floor((xp - R.min)/R.dx);      % # of nodes from the start of the current region
      node = n + n_offset;                  % add an offset to the local node number
%     fprintf( 'region: %g, n: %g, node:%g, xp: %16.15E dx: %g R.min: %16.15E , R.max: %16.15E  n_offset: %g\n',r, n, node, xp, dx, R.min, R.max,n_offset);
      return;
    end

    n_offset = (n_offset) + R.NN;           % increment the offset
  end
  
  %bulletproofing
  if( xp < Regions{1}.min || xp > Regions{nRegions}.max)
    fprintf( 'ERROR: positionToNode(), the particle (xp: %g) is outside the computational domain( %g, %g )\n',xp,Regions{1}.min,Regions{nRegions}.max  );
    input('stop'); 
  end
 
  if( node == 0)
    fprintf( 'ERROR: positionToNode(): node index must be > 0 \n');
  end
end


%__________________________________
function[nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos)
  [node, dx]=positionToNode(xp, nRegions, Regions);
  
  relativePosition = abs((xp) - nodePos(node))/dx;
  
  offset = int32(0);
  if( relativePosition < 0.5)
    offset = -1;
  end
  
  % bulletproofing
  if(relativePosition< 0 )
    fprintf('ERROR: positionToClosestNodes, relative position < 0 \n');
    fprintf( 'Node %g, offset :%g relative Position: %g, xp:%g, nodePos:%g \n',node, offset, relativePosition,xp, nodePos(node));
    input('stop');
  end
  
  nodes(1) = node + offset;
  nodes(2) = nodes(1) + 1;
  nodes(3) = nodes(2) + 1;
  %fprintf( 'xp:%g, node(1):%g, node(2):%g, node(3):%g relativePosition:%g\n',xp, nodes(1), nodes(2), nodes(3), relativePosition);
end
%__________________________________
% returns the initial volP and lp
function[volP_0, lp_0]=positionToVolP(xp, nRegions, Regions)
  volP_0 = -9.0;
  lp_0 = -9.0;
 
  for r=1:nRegions
    R = Regions{r};
    if ( (xp >= R.min) && (xp < R.max) )
      volP_0 = R.dx;
      lp_0   = R.lp;
    end
  end
end


%__________________________________
%  Equation 14 of "Structured Mesh Refinement in Generalized Interpolation Material Point Method
%  for Simulation of Dynamic Problems"
function [nodes,Ss]=findNodesAndWeights_linear(xp, notused, nRegions, Regions, nodePos, Lx)
  global NSFN;
  % find the nodes that surround the given location and
  % the values of the shape functions for those nodes
  % Assume the grid starts at x=0.  This follows the numenclature
  % of equation 12 of the reference

  [node, dx]=positionToNode(xp, nRegions, Regions);  

  nodes(1)= node;
  nodes(2)= node+1;
  
  for ig=1:NSFN
    Ss(ig) = -9;
    
    Lx_minus = Lx(nodes(ig),1);
    Lx_plus  = Lx(nodes(ig),2);
    delX = xp - nodePos(nodes(ig));

    if (delX <= -Lx_minus)
      Ss(ig) = 0;
    elseif(-Lx_minus <= delX && delX<= 0.0)
      Ss(ig) = 1.0 + delX/Lx_minus;
    elseif(  0 <= delX && delX<= Lx_plus)
      Ss(ig) = 1.0 - delX/Lx_plus;
    elseif( Lx_plus <= delX )
      Ss(ig) = 0;
    end
  end
  %__________________________________
  % bullet proofing
  % is the particle's position exactly coincidental with a node
  tst = 1.0./Ss;
  isinf(tst);

  if( isinf(tst(1)) || isinf(tst(2)) )
    fprintf('\n\nWARNING:__________________________________\n');
    fprintf(' The particle is positioned exactly on the node, xp: %16.15E  node: %g \n',xp, node);
    fprintf(' Ss(1): %16.15E    Ss(2): %16.15E \n',Ss(1), Ss(2));
    fprintf(' abs(xp - node(1))/dx: %16.15E\n',abs(xp - nodePos(nodes(1)) )/dx );
    fprintf(' now adding/subtracting fuzz to the weights\n');
    
    fuzz = 1e-15;
    if(Ss(1) == 0.0)
      Ss(1) = Ss(1) + fuzz;
      Ss(2) = Ss(2) - fuzz; 
    end
    if(Ss(2) == 0.0)
      Ss(2) = Ss(2) + fuzz;
      Ss(1) = Ss(1) - fuzz; 
    end
    fprintf(' Modified:Ss(1): %16.15E    Ss(2): %16.15E \n\n',Ss(1), Ss(2));
  end

  %  Do the shape functions sum to 1.0 
  sum = double(0);
  for ig=1:NSFN
    sum = sum + Ss(ig);
  end
  if ( abs(sum-1.0) > 1e-10)
    fprintf('findNodesAndWeights_linear\n');
    fprintf('node(1):%g, node(2):%g, xp:%g Ss(1): %g, Ss(2): %g, sum: %g\n',nodes(1),nodes(2), xp, Ss(1), Ss(2), sum)
    input('error: the shape functions (linear) dont sum to 1.0 \n');
  end

end


%__________________________________
%  Reference:  Uintah Documentation Chapter 7 MPM, Equation 7.16
function [nodes,Ss]=findNodesAndWeights_gimp(xp, lp, nRegions, Regions, nodePos, notUsed)
  global NSFN;

 [nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos);
  
  L = dx;
  
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
  
  %__________________________________
  % bullet proofing
  sum = double(0);
  for ig=1:NSFN
    sum = sum + Ss(ig);
  end
  if ( abs(sum-1.0) > 1e-10)
    fprintf('findNodesAndWeights_gimp\n');
    fprintf('delX: %g, node(1):%g, node(2):%g ,node(3):%g, xp:%g Ss(1): %g, Ss(2): %g, Ss(3): %g, sum: %g\n',delX,nodes(1),nodes(2),nodes(3), xp, Ss(1), Ss(2), Ss(3), sum)
    input('error: the shape functions dont sum to 1.0 \n');
  end
  
end

%__________________________________
%  Equation 15 of "Structured Mesh Refinement in Generalized Interpolation Material Point Method
%  for Simulation of Dynamic Problems"
function [nodes,Ss]=findNodesAndWeights_gimp2(xp, lp, nRegions, Regions, nodePos, Lx)

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

    delX = xp - nodePos(node);
    A = delX - lp;
    B = delX + lp;
    a = max( A, -Lx_minus);
    b = min( B,  Lx_plus);
    
    if (B <= -Lx_minus || A >= Lx_plus)
      
      Ss(ig) = 0;
      tmp = 0;
    elseif( b <= 0 )
    
      t1 = b - a;
      t2 = (b*b - a*a)/(2.0*Lx_minus);
      Ss(ig) = (t1 + t2)/(2.0*lp);
      
      tmp = (b-a+(b*b-a*a)/2/Lx_minus)/2/lp;
    elseif( a >= 0 )
      
      t1 = b - a;
      t2 = (b*b - a*a)/(2.0*Lx_plus);
      Ss(ig) = (t1 - t2)/(2.0*lp);
      
      tmp = (b-a-(b*b-a*a)/2/Lx_plus)/2/lp;
    else
    
      t1 = b - a;
      t2 = (a*a)/(2.0*Lx_minus);
      t3 = (b*b)/(2.0*Lx_plus);
      Ss(ig) = (t1 - t2 - t3)/(2*lp);
      
      tmp = (-a-a*a/2/Lx_minus+b-b*b/2/Lx_plus)/2/lp;
    end
    
    if( abs(tmp - Ss(ig)) > 1e-13)
      fprintf(' Ss: %g  tmp: %g \n', Ss(ig), tmp);
      fprintf( 'Node: %g xp: %g nodePos: %g\n', nodes(ig), xp, nodePos(node));
      fprintf( 'A: %g B: %g, a: %g, b: %g Lx_minus: %g, Lx_plus: %g lp: %g\n', A, B, a, b, Lx_minus, Lx_plus,lp);
      fprintf( '(B <= -Lx_minus || A >= Lx_plus) :%g \n',(B <= -Lx_minus || A >= Lx_plus));
      fprintf( '( b <= 0 ) :%g \n',( b <= 0 ));
      fprintf( '( a >= 0 ) :%g \n',( a >= 0 )); 
      input('error shape functions dont match\n');
    end
  end
  

  
  %__________________________________
  % bullet proofing
  sum = double(0);
  for ig=1:NSFN
    sum = sum + Ss(ig);
  end
  if ( abs(sum-1.0) > 1e-10)
    fprintf('findNodesAndWeights_gimp2 \n');
    fprintf('node(1):%g, node(1):%g ,node(3):%g, xp:%g Ss(1): %g, Ss(2): %g, Ss(3): %g, sum: %g\n',nodes(1),nodes(2),nodes(3), xp, Ss(1), Ss(2), Ss(3), sum)
    input('error: the shape functions dont sum to 1.0 \n');
  end
  
  %__________________________________
  % error checking
  % Only turn this on with single resolution grids
  if(0)
  [nodes,Ss_old]=findNodesAndWeights_gimp(xp, nRegions, Regions, nodePos, Lx);
  for ig=1:NSFN
    if ( abs(Ss_old(ig)-Ss(ig)) > 1e-10 )
      fprintf(' The methods (old/new) for computing the shape functions dont match\n');
      fprintf('Node: %g, Ss_old: %g, Ss_new: %g \n',node(ig), Ss_old(ig), Ss(ig));
      input('error: shape functions dont match \n'); 
    end
  end
  end
end

%__________________________________
%  Reference:  Uintah Documentation Chapter 7 MPM, Equation 7.14
function [nodes,Gs, dx]=findNodesAndWeightGradients_linear(xp, notUsed, nRegions, Regions, nodePos, notUsed2)
 
  % find the nodes that surround the given location and
  % the values of the gradients of the linear shape functions.
  % Assume the grid starts at x=0.

  [node, dx]=positionToNode(xp,nRegions, Regions);

  nodes(1) = node;
  nodes(2) = nodes(1)+1;

  Gs(1) = -1/dx;
  Gs(2) = 1/dx;
end

%__________________________________
%  Reference:  Uintah Documentation Chapter 7 MPM, Equation 7.17
function [nodes,Gs, dx]=findNodesAndWeightGradients_gimp(xp, lp, nRegions, Regions, nodePos,notUsed)

  global NSFN;
  % find the nodes that surround the given location and
  % the values of the gradients of the shape functions.
  % Assume the grid starts at x=0.  
  [nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos);
  
  L  = dx;
  
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
  
  %__________________________________
  % bullet proofing
  sum = double(0);
  for ig=1:NSFN
    sum = sum + Gs(ig);
  end
  if ( abs(sum) > 1e-10)
    fprintf('findNodesAndWeightGradients_gimp \n');
    fprintf('delX:%g node(1):%g, node(1):%g ,node(3):%g, xp:%g Gs(1): %g, Gs(2): %g, Gs(3): %g, sum: %g\n',delX,nodes(1),nodes(2),nodes(3), xp, Gs(1), Gs(2), Gs(3), sum)
    input('error: the gradient of the shape functions (gimp) dont sum to 1.0 \n');
  end
end

%__________________________________
%  The equations for this function are derived in the hand written notes.  
% The governing equations for the derivation come from equation 15.
function [nodes,Gs, dx]=findNodesAndWeightGradients_gimp2(xp, lp, nRegions, Regions, nodePos,Lx)

  global NSFN;
  
  [nodes,dx]=positionToClosestNodes(xp,nRegions,Regions, nodePos);
  
  for ig=1:NSFN
    Gs(ig) = -9;
      
    node = nodes(ig);                                                       
    Lx_minus = Lx(node,1);                                                  
    Lx_plus  = Lx(node,2);                                                  

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

        Gs(ig) = (-Lx_minus - A)/(2.0 * Lx_minus * lp);

      elseif( (B < Lx_plus) && (A <= -Lx_minus) )

        Gs(ig) = (Lx_plus - B)/(2.0 * Lx_plus * lp);

      else
        Gs(ig) = 0.0;
      end
    end                                          
  end
  
  %__________________________________
  % bullet proofing
  sum = double(0);
  for ig=1:NSFN
    sum = sum + Gs(ig);
  end
  if ( abs(sum) > 1e-10)
    fprintf('findNodesAndWeightGradients_gimp2 \n');
    fprintf('node(1):%g, node(2):%g ,node(3):%g, xp:%g Gs(1): %g, Gs(2): %g, Gs(3): %g, sum: %g\n',nodes(1),nodes(2),nodes(3), xp, Gs(1), Gs(2), Gs(3), sum)
    input('error: the gradient of the shape functions (gimp2) dont sum to 0.0 \n');
  end
end
end
