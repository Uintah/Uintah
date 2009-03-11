   
  //  ************************ benchmark case **********************
  
  // benchmark case
    int fakeIndex = 0;
  double xx, yy, zz;
  for ( int k = 0; k < Ncz; k ++ ){
    for ( int j = 0; j < Ncy; j ++) {
      for ( int i = 0; i < Ncx; i ++ ) {
	
	T_Vol[fakeIndex] = 64.80721904; // k
	xx = (X[i] + X[i+1])/2;
	yy = (Y[j] + Y[j+1])/2;
	zz = (Z[k] + Z[k+1])/2;

	kl_Vol[fakeIndex] = 0.9 * ( 1 - 2 * abs ( xx ) )
	  * ( 1 - 2 * abs ( yy ) )
	  * ( 1 - 2 * abs ( zz ) ) + 0.1;
	
	a_Vol[fakeIndex] = 1;
	scatter_Vol[fakeIndex] = 0;
	fakeIndex++;
	
      }
    }
  }
  
  
  // with participating media, and all cold black surfaces around
  
  // top bottom surfaces
  for ( int i = 0; i < TopBottomNo; i ++ ) {
    rs_top_surface[i] = 0;
    rs_bottom_surface[i] = 0;

    rd_top_surface[i] = 0;
    rd_bottom_surface[i] = 0;
    
    alpha_top_surface[i] = 1 - rs_top_surface[i] - rd_top_surface[i];
    alpha_bottom_surface[i] = 1 - rs_bottom_surface[i] - rd_bottom_surface[i];
        
    emiss_top_surface[i] = alpha_top_surface[i];
    emiss_bottom_surface[i] = alpha_bottom_surface[i];

    T_top_surface[i] = 0;
    T_bottom_surface[i] = 0;

    a_top_surface[i] = 1;
    a_bottom_surface[i] = 1;
    
  }


  // front back surfaces
  for ( int i = 0; i < FrontBackNo; i ++ ) {
    rs_front_surface[i] = 0;
    rs_back_surface[i] = 0;

    rd_front_surface[i] = 0;
    rd_back_surface[i] = 0;
    
    alpha_front_surface[i] = 1 - rs_front_surface[i] - rd_front_surface[i];
    alpha_back_surface[i] = 1 - rs_back_surface[i] - rd_back_surface[i];
        
    emiss_front_surface[i] = alpha_front_surface[i];
    emiss_back_surface[i] = alpha_back_surface[i];

    T_front_surface[i] = 0;
    T_back_surface[i] = 0;

    a_front_surface[i] = 1;
    a_back_surface[i] = 1;

  }

  
  // from left right surfaces
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    rs_left_surface[i] = 0;
    rs_right_surface[i] = 0;

    rd_left_surface[i] = 0;
    rd_right_surface[i] = 0;
    
    alpha_left_surface[i] = 1 - rs_left_surface[i] - rd_left_surface[i];
    alpha_right_surface[i] = 1 - rs_right_surface[i] - rd_right_surface[i];
        
    emiss_left_surface[i] = alpha_left_surface[i];
    emiss_right_surface[i] = alpha_right_surface[i];

    T_left_surface[i] = 0;
    T_right_surface[i] = 0;

    a_left_surface[i] = 1;
    a_right_surface[1] = 1;
        
    
  }
  // *************************** end of benchmark case **********************   
  
