   
  //  ************************ benchmark case **********************
  
  // benchmark case
    int fakeIndex = 0;
  double xx, yy, zz;
  for ( int k = 0; k < Ncz; k ++ ){
    for ( int j = 0; j < Ncy; j ++) {
      for ( int i = 0; i < Ncx; i ++ ) {
	
	T_Vol[fakeIndex] = 500;
	//	T_Vol[fakeIndex] = 64.80721904; // k
	xx = (X[i] + X[i+1])/2;
	yy = (Y[j] + Y[j+1])/2;
	zz = (Z[k] + Z[k+1])/2;

// 	kl_Vol[fakeIndex] = 0.9 * ( 1 - 2 * abs ( xx ) )
// 	  * ( 1 - 2 * abs ( yy ) )
// 	  * ( 1 - 2 * abs ( zz ) ) + 0.1;

	kl_Vol[fakeIndex] = 5;
	a_Vol[fakeIndex] = 1;
	scatter_Vol[fakeIndex] = 0;
	fakeIndex++;
	
      }
    }
  }
  
  
  // with participating media, and all cold black surfaces around
  
  // top bottom surfaces
  for ( int i = 0; i < TopBottomNo; i ++ ) {
    rs_surface[TOP][i] = 0;
    rs_surface[BOTTOM][i] = 0;

    rd_surface[TOP][i] = 0;
    rd_surface[BOTTOM][i] = 0;
    
    alpha_surface[TOP][i] = 1 - rs_surface[TOP][i] - rd_surface[TOP][i];
    alpha_surface[BOTTOM][i] = 1 - rs_surface[BOTTOM][i] - rd_surface[BOTTOM][i];
        
    emiss_surface[TOP][i] = alpha_surface[TOP][i];
    emiss_surface[BOTTOM][i] = alpha_surface[BOTTOM][i];

    T_surface[TOP][i] = 0;
    T_surface[BOTTOM][i] = 0;

    a_surface[TOP][i] = 1;
    a_surface[BOTTOM][i] = 1;
    
  }


  // front back surfaces
  for ( int i = 0; i < FrontBackNo; i ++ ) {
    rs_surface[FRONT][i] = 0;
    rs_surface[BACK][i] = 0;

    rd_surface[FRONT][i] = 0;
    rd_surface[BACK][i] = 0;
    
    alpha_surface[FRONT][i] = 1 - rs_surface[FRONT][i] - rd_surface[FRONT][i];
    alpha_surface[BACK][i] = 1 - rs_surface[BACK][i] - rd_surface[BACK][i];
        
    emiss_surface[FRONT][i] = alpha_surface[FRONT][i];
    emiss_surface[BACK][i] = alpha_surface[BACK][i];

    T_surface[FRONT][i] = 0;
    T_surface[BACK][i] = 0;

    a_surface[FRONT][i] = 1;
    a_surface[BACK][i] = 1;

  }

  
  // from left right surfaces
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    rs_surface[LEFT][i] = 0;
    rs_surface[RIGHT][i] = 0;

    rd_surface[LEFT][i] = 0;
    rd_surface[RIGHT][i] = 0;
    
    alpha_surface[LEFT][i] = 1 - rs_surface[LEFT][i] - rd_surface[LEFT][i];
    alpha_surface[RIGHT][i] = 1 - rs_surface[RIGHT][i] - rd_surface[RIGHT][i];
        
    emiss_surface[LEFT][i] = alpha_surface[LEFT][i];
    emiss_surface[RIGHT][i] = alpha_surface[RIGHT][i];

    T_surface[LEFT][i] = 0;
    T_surface[RIGHT][i] = 0;

    a_surface[LEFT][i] = 1;
    a_surface[RIGHT][1] = 1;
        
    
  }
  // *************************** end of benchmark case **********************   
  
