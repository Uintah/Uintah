
// ====== Webb homogeneous case ;
//FSK Case1 ==20% H2O, 10% CO2, 3% CO at T= T0 = 1000K, P = 1atm
// === 1D slab, 0 to 1 m with cold black walls
     
    int fakeIndex = 0;
  for ( int k = 0; k < Ncz; k ++ ){
    for ( int j = 0; j < Ncy; j ++) {
      for ( int i = 0; i < Ncx; i ++ ) {
	
	T_Vol[fakeIndex] = 1000; // k

	a_Vol[fakeIndex] = 1;
	scatter_Vol[fakeIndex] = 0;
	fakeIndex++;
	
      }
    }
  }



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
// for FSK fix g

//int wgkaSize;
//wgkaSize = coluwgka * 4;
//double *wgka = new double[wgkaSize];

//ToArray(wgkaSize,wgka,"wgkaAlpha25EvenrankWebbhomo");


// for R CDF with g r wavenumber

// read in Random number with g and k
cout << "read in data" << endl;
// Rg
//ToArray(gSize, Rkg, "afterInterpolationRkg"); // get Rkg from file
//ToArray(gkSize, gk, "HITEMPoldLBLkgT1000Trad1000-CO201H2O02CO003.dat");

// Reta
ToArray(gSize, Rkg, "RwvnabcsNoIb.dat"); // get Rwvn --no planck function weighting

//ToArray(gSize, Rkg, "RwvnabcsNosorting2.dat"); // get Rwvn -- CDF obtained from planck function weighted
ToArray(gkSize, gk, "LBLabsc-wvnm-T1000Trad1000-CO201H2O02CO003.dat"); // get abcswvnm
