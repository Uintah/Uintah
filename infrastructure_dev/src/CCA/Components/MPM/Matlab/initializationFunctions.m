function [IF] = initializationFunctions

  % create function handles that are used in AMRMPM.m
  IF.initialize_Regions     = @initialize_Regions;
  IF.initialize_NodePos     = @initialize_NodePos;
  IF.initialize_Lx          = @initialize_Lx;
  IF.initialize_xp          = @initialize_xp;
 
  %______________________________________________________________________
  function[nodePos]  = initialize_NodePos(NN, R1_dx, Regions, nRegions, interpolation)
    nodePos = zeros(NN,1);              % node Position
    nodeNum = int32(1);

    nodePos(1) = Regions{1}.min;

    for r=1:nRegions
      R = Regions{r};
      % loop over all nodes and set the node position
      
      if(r > 1)
        nodePos(nodeNum+1) = R.min;
      end
      
      while((nodePos(nodeNum) + R.dx) <= R.max)
        nodeNum = nodeNum +1;
        nodePos(nodeNum) = nodePos(nodeNum-1) + R.dx;
      end
    end
  end
  
  %______________________________________________________________________
  function[Lx]  = initialize_Lx(NN, nodePos)
    Lx      = zeros(NN,2);
    % compute the zone of influence
    Lx(1,1)  = 0.0;
    Lx(1,2)  = nodePos(2) - nodePos(1);
    Lx(NN,1) = nodePos(NN) - nodePos(NN-1);
    Lx(NN,2) = 0.0;

    for n=2:NN-1
      Lx(n,1) = nodePos(n) - nodePos(n-1);
      Lx(n,2) = nodePos(n+1) - nodePos(n);
    end
  end

  %______________________________________________________________________
  function[xp, NP] = initialize_xp(NN, nodePos, interpolation, PPC, bar_min, bar_max)
     %__________________________________
     % create particles
     fprintf('Particle Position\n');
     ip = 1;

     startNode = 1;
     if( strcmp(interpolation,'GIMP') )
       startNode = 2;
     end

     for n=startNode:NN-1
       dx_p = (nodePos(n+1) - nodePos(n) )/double(PPC);

       offset = dx_p/2.0;

       for p = 1:PPC
         xp_new = nodePos(n) + double(p-1) * dx_p + offset;

         if( xp_new >= bar_min && xp_new <= bar_max)

           xp(ip) = xp_new;

           fprintf('nodePos: %4.5e \t xp(%g) %g \t dx_p: %g \t offset: %g',nodePos(n),ip, xp(ip),dx_p,offset);

           if(ip > 1)
             fprintf( '\t \tdx: %g \n',(xp(ip) - xp(ip-1)));
           else
             fprintf('\n');
           end

           ip = ip + 1;

         end
       end
     end


     NP=ip-1;  % number of particles

     xp = reshape(xp, NP,1);  
  end
  %______________________________________________________________________
  function [Regions, nRegions,NN, dx_min] = initialize_Regions(domain,PPC,R1_dx,interpolation,d_smallNum)
    %__________________________________
    % region structure 
    %
    if(0)
    fprintf('USING plotShapeFunction regions\n');

    nRegions      = int32(1);              % partition the domain into numRegions
    Regions       = cell(nRegions,1);      % array that holds the individual region information
    R.refineRatio = 1;

    R.min         = -1;                     % location of left node
    R.max         = 1;                     % location of right node
    R.dx          = 1;
    R.NN          = int32( (R.max - R.min)/R.dx +1 ); % number of nodes interior nodes
    R.lp          = R.dx/(2 * PPC);
    Regions{1}    = R;
    end


    if(0)
    %____________
    % single level
    nRegions    = int32(2);               % partition the domain into nRegions
    Regions       = cell(nRegions,1);     % array that holds the individual region information
    R.min         = 0;                    % location of left point
    R.max         = domain/2;             % location of right point
    R.refineRatio = 1;
    R.dx          = R1_dx;
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx +1 );
    R.lp          = R.dx/(2 * PPC);
    Regions{1}    = R;

    R.min         = domain/2;                       
    R.max         = domain;
    R.refineRatio = 1;
    R.dx          = R1_dx/R.refineRatio;
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx );
    R.lp          = R.dx/(2 * PPC);
    Regions{2}    = R;
    end
    %____________
    % 2 level
    if(1)
    nRegions    = int32(3);               % partition the domain into nRegions
    Regions       = cell(nRegions,1);     % array that holds the individual region information

    R.min         = 0;                    % location of left point
    R.max         = domain/3.0;             % location of right point
    R.refineRatio = 1;
    R.dx          = R1_dx;
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx );
    R.lp          = R.dx/(2 * PPC);
    Regions{1}    = R;

    R.min         = domain/3.0;                       
    R.max         = 2.0*domain/3.0;
    R.refineRatio = 2;
    R.dx          = R1_dx/R.refineRatio;
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx );
    R.lp          = R.dx/(2 * PPC);
    Regions{2}    = R;

    R.min         = 2.0*domain/3.0;                       
    R.max         = domain;
    R.refineRatio = 1;
    R.dx          = R1_dx/R.refineRatio; 
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx +1); 
    R.lp          = R.dx/(2 * PPC);
    Regions{3}    = R;

    end

    %____________
    % 3 levels
    if(0)

    nRegions    = int32(5);               % partition the domain into nRegions
    Regions       = cell(nRegions,1);     % array that holds the individual region information

    R.min         = 0;                    % location of left point
    R.max         = 0.32;                 % location of right point
    R.refineRatio = 1;
    R.dx          = R1_dx;
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx +1 );
    R.lp          = R.dx/(2 * PPC);
    Regions{1}    = R;

    R.min         = 0.32;                       
    R.max         = 0.4;
    R.refineRatio = 4;
    R.dx          = R1_dx/double(R.refineRatio);
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx );
    R.lp          = R.dx/(2 * PPC);
    Regions{2}    = R;

    R.min         = 0.4;                       
    R.max         = 0.56;
    R.refineRatio = 16;
    R.dx          = R1_dx/double(R.refineRatio); 
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx);
    R.lp          = R.dx/(2 * PPC); 
    Regions{3}    = R;

    R.min         = 0.56;                       
    R.max         = 0.64;
    R.refineRatio = 4;
    R.dx          = R1_dx/double(R.refineRatio);
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx );
    R.lp          = R.dx/(2 * PPC);
    Regions{4}    = R;

    R.min         = 0.64;                       
    R.max         = domain;
    R.refineRatio = 1;
    R.dx          = R1_dx/R.refineRatio;
    R.volP        = R.dx/PPC;
    R.NN          = int32( (R.max - R.min)/R.dx );
    R.lp          = R.dx/(2 * PPC);
    Regions{5}    = R;

    end

    % increase the number of nodes in the first and last region if using gimp.
    if(strcmp(interpolation,'GIMP'))
      Regions{1}.NN          = Regions{1}.NN + 1;
      Regions{1}.min         = Regions{1}.min - Regions{1}.dx;
      Regions{nRegions}.NN   = Regions{nRegions}.NN + 1;
      Regions{nRegions}.max  = Regions{nRegions}.max + Regions{nRegions}.dx;
    end;

    % Define the extra cells L & R for each region.
    for r=1:nRegions
      Regions{r}.EC(1) = 0;    
      Regions{r}.EC(2) = 0;    
    end

    if(strcmp(interpolation,'GIMP'))
      Regions{1}.EC(1)        = 1;
      Regions{nRegions}.EC(2) = 1;
    end;

    %count the number of total nodes
    NN = int32(0);
    for r=1:nRegions
      R = Regions{r};
      NN = NN + R.NN;
    end
    
    %  find the minimum dx
    dx_min = double(1e100);
    for r=1:nRegions
      R = Regions{r};
      dx_min = min(dx_min,R.dx);
      fprintf( 'region %g, min: %g, \t max: %g \t refineRatio: %g dx: %g, NN: %g\n',r, R.min, R.max, R.refineRatio, R.dx, R.NN)
    end

    % bulletproofing:
    for r=1:nRegions
      R = Regions{r};
      d = (R.max - R.min) + 100* d_smallNum;

      if( mod( d, R.dx ) > 1.0e-10 )
        fprintf('ERROR, the dx: %g in Region %g does not divide into the domain (R.max:%g R.min:%g) evenly\n', R.dx,r,R.max,R.min);
        return;
      end
    end

  end
  
end
