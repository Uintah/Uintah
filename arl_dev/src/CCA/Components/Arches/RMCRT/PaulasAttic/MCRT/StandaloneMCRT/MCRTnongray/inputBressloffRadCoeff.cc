// ====== Case 6 in Chapter 3 =====
     
//    double *CO2 = new double [VolElementNo];
//    double *H2O = new double [VolElementNo];
//    double *SFV = new double [VolElementNo];

   // and we already have T_Vol for control volumes.

   int i_index;
   double xaveg;
   // as the properties only change with x, so calculate x's first
   // then simply assign these values to ys and zs.

// configure A soot = 1e-7 magnitude
// configure B soot = 1e-8 magnitude
   for ( int i = 0; i < Ncx; i ++ ) {
     
     xaveg = ( X[i] + X[i+1] + Lx )/2;
     CO2[i] = 0.4 * xaveg * ( 1 - xaveg ) + 0.06;
     H2O[i] = 2 * CO2[i];
     SFV[i] = ( 40 * xaveg * ( 1 - xaveg) + 6 ) * 1e-7;
     T_Vol[i] = 4000 * xaveg * ( 1 - xaveg ) + 800;

     // for all ys and zs
     for ( int m =  0; m < Ncz; m ++ ) {
       for ( int n = 0; n < Ncy; n ++ ) {
	 i_index = i + Ncx * n + TopBottomNo * m;
	 CO2[i_index] = CO2[i];
	 H2O[i_index] = H2O[i];
	 SFV[i_index] = SFV[i];
	 T_Vol[i_index] = T_Vol[i];
       }
     }

     
   }


   double OPL;
   OPL = 1.76;

   // cout << " I am here now after define CO2, H2O, SFV, and T_Vol line 802 " << endl;
   
   RadCoeff obRadCoeff(OPL);

   
for ( int i = 0; i < VolElementNo; i ++ ){
     scatter_Vol[i] = 0;
     a_Vol[i] = 1;
 }

// only one band , one coefficient,
// thus can be done fully in input.cc file
   obRadCoeff.PrepCoeff(CO2, H2O, SFV, T_Vol, kl_Vol,
			VolElementNo, TopBottomNo,
			Ncx, Ncy, Ncz);
   

   
   // making the font, back, top bottom surfaces as mirrors
   // so the left and right surfaces would be infinite big.
// thus property changes along X

  // top bottom surfaces
  for ( int i = 0; i < TopBottomNo; i ++ ) {
    rs_surface[TOP][i] = 1;
    rs_surface[BOTTOM][i] = 1;

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
    rs_surface[FRONT][i] = 1;
    rs_surface[BACK][i] = 1;

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

   // ============= end of Case 6 in chapter 3 ===
   
  
