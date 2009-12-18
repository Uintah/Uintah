function [Regions, nRegions,NN] = initializeRegions(domain,PPC,R1_dx,interpolation,d_smallNum)
  %__________________________________
  % region structure 
  %
  if(0)
  fprintf('USING plotShapeFunction regions\n');
  
  nRegions      = int32(1);              % partition the domain into numRegions
  Regions       = cell(nRegions,1);      % array that holds the individual region information

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
  R.NN          = int32( (R.max - R.min)/R.dx +1 );
  R.lp          = R.dx/(2 * PPC);
  Regions{1}    = R;

  R.min         = domain/3.0;                       
  R.max         = 2.0*domain/3.0;
  R.refineRatio = 1;
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
  R.NN          = int32( (R.max - R.min)/R.dx); 
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

  NN = int32(0);
  for r=1:nRegions
    R = Regions{r};
    NN = NN + R.NN;
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
