
// ====== Webb non-homogeneous case ;
//FSK  ==20% H2O, 10% CO2, 3% CO at T= T0 = 1000K, P = 1atm
// 3 layers == middle -- 40% H2O, 20% CO2, 6% CO at T= T0 = 1500K, P = 1atm
// === 1D slab, 0 to 1 m with cold black walls

// layers lie along x direction

   // Webb case , 3 layers in z direction; with top and bottom at 1000K black.
   int xI, xII; // indices for x
   int xi;
   
   xi = -1;
   double firstlayer = -Lx/2.0 + 0.35;
   double secondlayer = -Lx/2.0 + 0.65;
   double thirdlayer = -Lx/2.0 + 1.0;


     // Webb case , 3 layers in x direction; with top and bottom at 1000K e=0.8.
     
     do {
       xi++;
     }while( (X[xi]+X[xi+1])/2.0 < firstlayer );
     
     xI = xi;
     cout << " xI = " << xI << endl;
     // note: dont need to do xi -1 for finding the firstlayer index  xI
     xi--;
     do {
       xi++;
     }while( (X[xi]+X[xi+1])/2.0 < secondlayer );     

// xII = xi-1;
    xII = xi;
     cout << " xII = " << xII << endl;
     // note : need to do the xi - 1 for finding the secondlayer index xII

     // you will realize that when u take a look at the Xtable :)

  for ( int k = 0; k < Ncz; k ++ )
    for ( int j = 0; j < Ncy; j ++) 
      for ( int i = 0; i < xI; i ++ ) 
	T_Vol[k*Ncx*Ncy + j*Ncx +i] = 1000;


  for ( int k = 0; k < Ncz; k ++ )
    for ( int j = 0; j < Ncy; j ++) 
      for ( int i = xI; i < xII; i ++ ) 
	T_Vol[k*Ncx*Ncy + j*Ncx +i] = 1500;

  for ( int k = 0; k < Ncz; k ++ )
    for ( int j = 0; j < Ncy; j ++) 
      for ( int i = xII; i < Ncx; i ++ ) 
	T_Vol[k*Ncx*Ncy + j*Ncx +i] = 1000;

for ( int i = 0; i < Ncx; i ++ )
  {
    int k = int (floor(Ncz/2));
    int j = int (floor(Ncy/2));
    cout << "i = " << i << "TVol = " << T_Vol[k*TopBottomNo + j*Ncx + i] << endl;
  }


for ( int i = 0; i < VolElementNo; i ++){
  a_Vol[i] = 1;
  scatter_Vol[i] = 0;
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

    emiss_surface[LEFT][i] = 0.8;
    emiss_surface[RIGHT][i] = 0.8;
    
    rs_surface[LEFT][i] = 0;
    rs_surface[RIGHT][i] = 0;
    
    alpha_surface[LEFT][i] =  emiss_surface[LEFT][i];
    alpha_surface[RIGHT][i] = emiss_surface[RIGHT][i];
        
    rd_surface[LEFT][i] = 1 - rs_surface[LEFT][i] - emiss_surface[LEFT][i];
    rd_surface[RIGHT][i] = 1 - rs_surface[RIGHT][i] - emiss_surface[RIGHT][i];
    
    T_surface[LEFT][i] = 1000;
    T_surface[RIGHT][i] = 1000;

    a_surface[LEFT][i] = 1;
    a_surface[RIGHT][1] = 1;
        
    
  }


cout << "read in data" << endl;

// Reta
// get data for T=1000 K section
ToArray(gSize, Rkgcold, "RwvnabcsNoIb.dat");
ToArray(gkSize, gkcold, "LBLabsc-wvnm-T1000Trad1000-CO201H2O02CO003.dat"); // get wvnm-abcs

// get data for T=1500 K section
ToArray(gSize, Rkghot, "RwvnabcsNoIb1500K-CO202H2O04CO006.dat"); // get Rwvn -- CDF obtained from planck function weighted
ToArray(gkSize, gkhot, "LBLabsc-wvnm-T1500Trad1500-CO202H2O04CO006.dat"); // get wvnm-abcs


  // make Rcold not starting from 0, as R = 0-> g= 0, -> small k, not important rays
  for ( int i = 0; i < Rcoldsize; i ++ ){
    Rcold[i] = (i+1) * dRcold;
    // cout << "Rcold = " << Rcold[i] << endl;
    Rgg = Rcold[i];
    // Since Rcold is pre-set, we can pre-calculate g and kl from Rcold.    
    obBST.search(Rgg, Rkgcold, gSize);
    obBST.calculate_gk(gkcold, Rkgcold, Rgg);
    gcold[i] = obBST.get_g(); // for Reta, convert wavenumber unit from cm-1 to m-1
    //  cout << " g= "<< g << endl;
    klcold[i] = obBST.get_k()*100; // convert unit from cm-1 to m-1
    Ibetacold[i] = C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / Tcold )- 1) / pi;
    
  }


  // make Rcold not starting from 0, as R = 0-> g= 0, -> small k, not important rays
  for ( int i = 0; i < Rhotsize; i ++ ){
    Rhot[i] = (i+1) * dRhot;
    // cout << "Rcold = " << Rcold[i] << endl;
    Rgg = Rhot[i];
    // Since Rcold is pre-set, we can pre-calculate g and kl from Rcold.    
    obBST.search(Rgg, Rkghot, gSize);
    obBST.calculate_gk(gkhot, Rkghot, Rgg);
    ghot[i] = obBST.get_g(); // for Reta, convert wavenumber unit from cm-1 to m-1
    //  cout << " g= "<< g << endl;
    klhot[i] = obBST.get_k()*100; // convert unit from cm-1 to m-1
    Ibetahot[i] = C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / Thot )- 1) / pi;
    
  }

  
    obTable.singleArrayTable(T_Vol, VolElementNo, 1, "TVolTablelast.dat");
   obTable.singleArrayTable(X, Npx, 1, "Xtable.dat");
    obTable.singleArrayTable(klhot, Rhotsize, 1, "abcsTableReta5000NoIbCDFhot.dat");
   obTable.singleArrayTable(ghot, Rhotsize, 1, "wvnTableReta5000NoIbCDFhot.dat");
   obTable.singleArrayTable(Rhot, Rhotsize, 1, "RhotTableReta5000NoIbCDFhot.dat");


// pre-processing IntenArray_IetahotSurface, IntenArray_IetacoldSurface.
/*
// top bottom surface
for ( int i = 0; i < TopBottomNo; i ++ ) {
  
  IntenArray_IetahotSurface[TOP][i] =  emiss_surface[TOP][i] *
    a_surface[TOP][i] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / T_surface[TOP][i] )- 1) / pi;
  
  IntenArray_IetacoldSurface[TOP][i] =  emiss_surface[TOP][i] *
    a_surface[TOP][i] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / T_surface[TOP][i] )- 1) / pi;

  
    IntenArray_IetahotSurface[BOTTOM][i] =  emiss_surface[BOTTOM][i] *
    a_surface[BOTTOM][i] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / T_surface[BOTTOM][i] )- 1) / pi;
  
  IntenArray_IetacoldSurface[BOTTOM][i] =  emiss_surface[BOTTOM][i] *
    a_surface[BOTTOM][i] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / T_surface[BOTTOM][i] )- 1) / pi;

 }


// front back surfaces
for ( int i = 0; i < FrontBackNo; i ++ ) {
  
  IntenArray_IetahotSurface[FRONT][i] =  emiss_surface[FRONT][i] *
    a_surface[FRONT][i] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / T_surface[FRONT][i] )- 1) / pi;
  
  IntenArray_IetacoldSurface[FRONT][i] =  emiss_surface[FRONT][i] *
    a_surface[FRONT][i] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / T_surface[FRONT][i] )- 1) / pi;

  
    IntenArray_IetahotSurface[BACK][i] =  emiss_surface[BACK][i] *
    a_surface[BACK][i] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / T_surface[BACK][i] )- 1) / pi;
  
  IntenArray_IetacoldSurface[BACK][i] =  emiss_surface[BACK][i] *
    a_surface[BACK][i] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / T_surface[BACK][i] )- 1) / pi;

 }



// left and right surfaces
for ( int i = 0; i < LeftRightNo; i ++ ) {
  
  IntenArray_IetahotSurface[LEFT][i] =  emiss_surface[LEFT][i] *
    a_surface[LEFT][i] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / T_surface[LEFT][i] )- 1) / pi;
  
  IntenArray_IetacoldSurface[LEFT][i] =  emiss_surface[LEFT][i] *
    a_surface[LEFT][i] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / T_surface[LEFT][i] )- 1) / pi;

  
    IntenArray_IetahotSurface[RIGHT][i] =  emiss_surface[RIGHT][i] *
    a_surface[RIGHT][i] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / T_surface[RIGHT][i] )- 1) / pi;
  
  IntenArray_IetacoldSurface[RIGHT][i] =  emiss_surface[RIGHT][i] *
    a_surface[RIGHT][i] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / T_surface[RIGHT][i] )- 1) / pi;

 }

*/





// hot section for surfaces
for ( int i = 0; i < rayNohot; i ++ ) {
  
  IntenArray_IetahotSurface[TOP][i] =  0;

  IntenArray_IetahotSurface[BOTTOM][i] =  0;
  
  IntenArray_IetahotSurface[FRONT][i] =  0;
  
  IntenArray_IetahotSurface[BACK][i] = 0;

  // only valid for a uniform surface
  IntenArray_IetahotSurface[LEFT][i] =  emiss_surface[LEFT][0] *
    a_surface[LEFT][0] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
      ( exp( C2*  ghot[i] / T_surface[LEFT][0] )- 1) / pi;

  IntenArray_IetahotSurface[RIGHT][i] =  emiss_surface[RIGHT][0] *
    a_surface[RIGHT][0] *
    C1 * ghot[i] *  ghot[i] *  ghot[i]  * 1e6 /
    ( exp( C2*  ghot[i] / T_surface[RIGHT][0] )- 1) / pi;
  
 }


// cold section for surfaces
// loop over all rayCount-- i.e. all Reta's eta.
for ( int i = 0; i < rayNocold; i ++ ) {
  IntenArray_IetacoldSurface[TOP][i] = 0;
  IntenArray_IetacoldSurface[BOTTOM][i] = 0;  
  IntenArray_IetacoldSurface[FRONT][i] =  0;  
  IntenArray_IetacoldSurface[BACK][i] = 0;
  
  IntenArray_IetacoldSurface[LEFT][i] =  emiss_surface[LEFT][0] *
    a_surface[LEFT][0] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
    ( exp( C2*  gcold[i] / T_surface[LEFT][0] )- 1) / pi;
  
  IntenArray_IetacoldSurface[RIGHT][i] =  emiss_surface[RIGHT][0] *
    a_surface[RIGHT][0] *
    C1 * gcold[i] *  gcold[i] *  gcold[i]  * 1e6 /
      ( exp( C2*  gcold[i] / T_surface[RIGHT][0] )- 1) / pi;  

 }

