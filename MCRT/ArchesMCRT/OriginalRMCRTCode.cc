//------- BackwardMCRTSolver.cc-----
// ------ Backward (Reverse ) Monte Carlo Ray-Tracing Radiation Model------
#include "mpi.h"
#include "RNG.h"
#include "Surface.h"
#include "Consts.h"
#include "RealSurface.h"
#include "TopRealSurface.h"
#include "BottomRealSurface.h"
#include "FrontRealSurface.h"
#include "BackRealSurface.h"
#include "LeftRealSurface.h"
#include "RightRealSurface.h"
#include "VirtualSurface.h"
#include "ray.h"
#include "MakeTableFunction.h"
#include "flux.h"
#include "VolElement.h"
#include "RadCoeff.h"
#include "KGA.h"


#include <cmath>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>

using namespace std;

// Parallel by Regions

// will face load-balancing problems when apply to nonhomogeneous media,
// expecially optical thick non-homogeneous media

// Backward MCRT starts from the termination point r_s, backward in path:

// LeftIntenFrac-- the fraction of left intensity emitted from the current position
//                 travelled to the termination point r_s

// previousSum -- the previous iteration of the total optical thickness
//                extinc_medium =   (scat_m + kl_m) * sumScat
// currentSum -- the current total optical thickness

// SurLeft -- the left percentage of real surface for intensity

// PathLeft --- the left percentage of intensity during the path in a control volume

// PathSurfaceLeft -- the left percentage of intensity from a REAL surface

// PathIndex -- tracking index along the path

// the ReverseMCRT.cc is the last updated program



void ToArray(int size, double *array, char *_argv){

  ifstream in(_argv); // open table
  if (!in){
    cout << " cannot open " << _argv << endl;
    exit(1);
  }
  
  for ( int i =0; i < size; i ++ )
      in >> array[i];
 
   in.close();
}




// benchmark case absorption coefficient
double BenchMark_k(const double &xx, const double &yy, const double &zz){
  double k;
  k = 0.9 * ( 1 - 2 * abs ( xx ) ) * ( 1 - 2 * abs ( yy ) ) * ( 1 - 2 * abs ( zz ) ) + 0.1;
  //  k = 0.5;
  return k;
}


double MeshSize(int &Nchalf, double &Lhalf, double &ratio){
  double dcenter;
  double pp;
  pp = pow(ratio,Nchalf);
  //  cout << "Nchalf = " << Nchalf << endl;
  //  cout << "Lhalf = " << Lhalf << endl;
  //  cout << "ratio = " << ratio << endl;
  //  cout << "pp = " << pp << endl;
  //  cout << "======================== " << endl;
  
  dcenter = (1-ratio)*Lhalf/(1- pow(ratio,Nchalf));    
  return dcenter;
}



int main(int argc, char *argv[]){
    
  int my_rank; // rank of process
  int np; // number of processes
  double time1, time2;
  double precision;
  //  int INHOMOGENEOUS;


  // starting up MPI
  MPI_Init(&argc, &argv);
  MPI_Barrier(MPI_COMM_WORLD);
  
  precision = MPI_Wtick();
  
  time1 = MPI_Wtime();
  
  // Find out process rank
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Find out number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &np);


  
  // number of cells in each coordinates
  // make sure to pick the appropriate CubeLen to get integral xno, yno, zno

  int rayNoSurface, rayNoVol;
  //  char caseflag;
  //  double casePlates;
  //  int  mixtureKg, TablePoints;

  // the No. of rays based on rayNoSurface and rayNoVol,
  //but proportional to medium's absorption coefficient kl(m-1)
  //================= rayNo Array ====================
  // i was trying to set rayNoArray as a function of not only Volume and Element Area, but also
  // absorption coefficent kl, as if kl is large ( optically thick medium ), the ray ends very shortly
  // after it is being emitted, so with the same No. of ray, it will result in a larger statistic error.

  // however, if rayNo(kl, vol, area), means, i need to find a standard for kl to affect rayNo,
  // as kl usualy vary from 1e-6 to 100, if just simple proportionally to klcurrent/klmin, it will result
  // in huge ray No.s, which will be equivelent to set most of the field as maximum rayNo.

  // and also for nonhomogeneous medium, i need to calculate rayNo differently for every element.
  // and the maximum rayNo should changes with kl too,
  // but it is hard to set that, in nonhomogeneous media, which kl are we refering to?

  // and i just got the idea that, IntenFrac( the threshold of stopping criteria)
  //should be changing with kl too.
  // larger IntenFrac ( 1e-6 )  for optically thin media ( say, kl = 1e-6  )
  // smaller IntenFrac ( 1e-15) for optically thick media ( say , kl = 10 ) 
  // but again for nonhomegeneous media, it is really hard to do so, as the ray may travel from
  // optically thin regioin to optically thick region, then which kl are we referring to?
  
  // int rayNoSurfacekl, rayNoVolkl;

  
  // For non-uniform mesh
  // To make symmetric mesh domain, need inputs as following:
  // half length of domain; half number of cell on each length; ratio ( == boundary cell/center cell)
  // ratio < 1 , then boundary cell < center cell, obtains finer mesh on the boundary
  // ratio > 1, then boundary cell > center cell, obtains finer mesh in the center
  // xcenter is the center's cell's dimension
  //  r-- ratio ( boundary/center); L -- total length; N-- total number of cell
  // dxcenter -- size of center cell in x direction
  // i.e. dxcenter * sum ( 1 + r + r^2 + ... + r^(N/2-1) ) = L/2
  // so, dxcenter = L/2 / sum( 1 + r + r^2 + .. + r^(N/2-1))
  // Sn = a1*(1-q^n)/(1-q) 

  // Lhalf and dxcenter is just the distance, spacing, not the coordinate
  int Ncx, Ncy, Ncz; // number of cells in x, y, z directions
  int Npx, Npy, Npz; // number of grid points
  int Ncxhalf, Ncyhalf, Nczhalf;  // half number of the cells in cooresponding direction
  double Lx, Ly, Lz; // full length of full domain in x, y, z direction
  double Lxhalf, Lyhalf, Lzhalf; // half length of full domain in x,y,z direction
  double ratioBCx, ratioBCy, ratioBCz;  // ratioBC = Boundary cell/ center cell
  double dxcenter, dycenter, dzcenter; // center cells' size
  
  // starting Nos on Top --- Right Surfaces
  int TopStartNo, BottomStartNo, FrontStartNo, BackStartNo, LeftStartNo, RightStartNo; 
  int VolElementNo, TopBottomNo, FrontBackNo, LeftRightNo, sumElementNo;
  int totalElementNo, surfaceElementNo;
  double EnergyAmount; // set as customer self-set-up later
  int surfaceIndex, currentIndex;
  double sumIncomInten, aveIncomInten;  
  bool OnlySurfaceCenter, DFSK, FSSK, FSCK, INHOMOGENEOUS;
  double Lh, Lc, Thot, Tcold; 


  
  if ( my_rank == 0 ) {
    //cout << "Please enter rayNoSurface, rayNoVol, CubeLenx,Cubeleny, CubeLenz" << endl;
   //  cout << " Please enter rayNoSurface " << endl;
//     cin >> rayNoSurface; // = atoi ( argv[1] );

    
    rayNoSurface = 1000;
    
//     cout << "Please enter rayNoVol " << endl;
//     cin >> rayNoVol;  // = atoi (argv[2]);
    rayNoVol = 0;
    
    //    cout << "Please enter Lx, Ly, Lz ( full length of full domain in x, y, z direction)" << endl;
//     cin >> Lx;
//     cin >> Ly;
//     cin >> Lz;

    //    cout << "please enter Ncx, Ncy, Ncz ( number of cells in x, y, z direction)" << endl;
  //   cin >> Ncx;
//     cin >> Ncy;
//     cin >> Ncz;
    Ncx = 20;
    Ncy = 20;
    Ncz = 20;
    
    cout << "Please enter ratioBC = boundary cell/ center cell" << endl;
    // cin >> ratioBC;
    ratioBCx = 1;
    ratioBCy = 1;
    ratioBCz = 1;
    
    //     cout << " Please enter in caseflag 1, 2, 3, or 4 " << endl;
    //     cin >> caseflag; // char
    //    caseflag = 0;
    
    //     cout << " Please enter plates case " << endl;
    //     cin >> casePlates;
    //   casePlates = 33;
    
    //    mixtureKg = 1;
    //    TablePoints = 0; // use 12G
    INHOMOGENEOUS = 1; // homogeneous
    OnlySurfaceCenter = 1; // only calculate surface center elements
      
  } // if my_rank == 0


  
  MPI_Barrier (MPI_COMM_WORLD);
  
  MPI_Bcast(&rayNoSurface, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rayNoVol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //  MPI_Bcast(&Lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //  MPI_Bcast(&Ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //  MPI_Bcast(&Lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ncx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ncy, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ncz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ratioBCx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ratioBCy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
  MPI_Bcast(&ratioBCz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&OnlySurfaceCenter, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
  //  MPI_Bcast(&caseflag, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
  //  MPI_Bcast(&casePlates, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //  MPI_Bcast(&mixtureKg,1,MPI_INT ,0,MPI_COMM_WORLD);
  //  MPI_Bcast(&TablePoints,1,MPI_INT ,0,MPI_COMM_WORLD);
  MPI_Bcast(&INHOMOGENEOUS,1,MPI_INT,0,MPI_COMM_WORLD);
  

  MPI_Barrier (MPI_COMM_WORLD);

  DFSK = 0;
  FSSK = 0;
  FSCK = 1;
  Thot = 2000;
  Tcold = 300;

  Lh = 0.5; // m
  Lc = 0.1; // m
  Lx = Lh + Lc;
  Ly = Lh + Lc;
  Lz = Lh + Lc;
      
  // numbers of grid points
  Npx = Ncx + 1;
  Npy = Ncy + 1;
  Npz = Ncz + 1;
  
  
  Ncxhalf = Ncx / 2;
  Ncyhalf = Ncy / 2;
  Nczhalf = Ncz / 2;
  Lxhalf = Lx /2;
  Lyhalf = Ly/2;
  Lzhalf = Lz/2;
  
  VolElementNo = Ncx * Ncy * Ncz;
  TopBottomNo = Ncx * Ncy;
  FrontBackNo = Ncx * Ncz;
  LeftRightNo = Ncy * Ncz;
  
  // watch out that it has its order: top->bottom->front->back->left->right
  TopStartNo = VolElementNo;// cuz index starts from 0
  BottomStartNo = TopStartNo + TopBottomNo;
  FrontStartNo = BottomStartNo + TopBottomNo;
  BackStartNo = FrontStartNo + FrontBackNo;
  LeftStartNo = BackStartNo + FrontBackNo;
  RightStartNo = LeftStartNo + LeftRightNo;
  sumElementNo = RightStartNo + LeftRightNo; // this is the real number - 1
  
  surfaceElementNo = 2 * ( Ncx * Ncy + Ncx * Ncz + Ncy * Ncz );
  totalElementNo = VolElementNo + surfaceElementNo;
  
    
  // get coordinates arrays
  double *X = new double [Npx];
  double *Y = new double [Npy];
  double *Z = new double [Npz];
  double *dx = new double[Ncx]; // x cell's size dx
  double *dy = new double[Ncy]; // y cell's size dy
  double *dz = new double[Ncz]; // z cell's size dz
  
  // get property of real surface
  double *T_surface = new double [surfaceElementNo];
  double *absorb_surface = new double [surfaceElementNo];
  double *rs_surface = new double [surfaceElementNo];
  double *rd_surface = new double [surfaceElementNo];
  double *emiss_surface = new double [surfaceElementNo];
  double *a_surface = new double[surfaceElementNo];
  
  // get property of vol
  double *T_Vol = new double [VolElementNo];
  double *kl_Vol = new double [VolElementNo];
  double *scatter_Vol = new double [VolElementNo];
  double *a_Vol = new double[VolElementNo];
  int *rayNo = new int [totalElementNo];

  
  // Top and Bottom surfaces
  double minArea, minAreaTB, minAreaFB, minAreaLR;
  double minVol;
   
  double *ElementAreaTB = new double[TopBottomNo];
  double *ElementAreaFB = new double[FrontBackNo];
  double *ElementAreaLR = new double[LeftRightNo];
  double *ElementVol = new double[VolElementNo];

  int VolTableSize;
  int VolNeighborSize;
  
  VolTableSize = VolElementNo * 13;
  double *VolTableArray = new double [VolTableSize];
    
  VolNeighborSize = VolElementNo * 7;
  double *VolNeighborArray = new double [VolNeighborSize];

   
  MPI_Barrier (MPI_COMM_WORLD);

  MakeTableFunction ob;
  int powNo;
  double dxUni, dyUni, dzUni; // for uniform mesh in x, y, z direction
  int maxiRayNoSurface, maxiRayNoVol;
  // maxiRayNoSurface = 8000;
  //  maxiRayNoVol = 8000;
  

  // ========= set values or get values for array pointers =============
  // the center of the cube is at (0,0,0) in a cartesian coordinate
  // the orgin of the cube ( domain ) can be changed here easily
  
   X[0] = -Lx/2.; // start from left to right
   Y[0] = Ly/2.; // start from back to front
   Z[0] = Lz/2; // start from top to bottom
   X[Npx-1] = Lx/2;
   Y[Npy-1] = -Ly/2;
   Z[Npz-1] = -Lz/2;

   
  // x direction

  if ( ratioBCx != 1 ) { 

    dxcenter = MeshSize(Ncxhalf,Lxhalf,ratioBCx);
    for ( int i = 0; i < Ncxhalf; i++ ){
      powNo = Ncxhalf-1-i;
      dx[i] = dxcenter * pow( ratioBCx, powNo );
      dx[Ncx-i-1] = dx[i];
    }
   
    // dont use x[i] = f ( x[i-1] ) , will get fatal error when cubelen is not integer.    
    for ( int i = 1; i < Ncxhalf ; i ++ )
      {
	X[i] = X[i-1] + dx[i-1];
	X[Ncx-i] = X[Npx-i] - dx[i-1];	
      }
  }
  else if ( ratioBCx == 1 ) {
    
    dxUni = Lx / Ncx;
    for ( int i = 1; i < Npx ; i ++ )
      {
	dx[i-1] = dxUni;
	X[i] = X[0] + i * dx[i-1]; 
      }
    
  }
  
/*
  if ( my_rank == 0 ) {
    ob.singleArrayTable(X,Npx,1,"XtableWebbL1NonUniform");      
    ob.singleArrayTable(dx,Ncx,1,"dxtableWebbL1NonUniform");    
  }
*/

  // y direction

  if ( ratioBCy != 1 ) {
    
    dycenter = MeshSize(Ncyhalf,Lyhalf,ratioBCy);
    for ( int i = 0; i < Ncyhalf; i++ ) {
      dy[i] = dycenter * pow( ratioBCy, Ncyhalf-1-i );
      dy[Ncy-i-1] = dy[i];
    }
    
    for ( int i = 1; i < Ncyhalf; i ++ )
      {
	Y[i] = Y[i-1] - dy[i-1];
	Y[Ncy-i] = Y[Npy-i] + dy[i-1];
      }    
  }
  else if ( ratioBCy == 1 ) {
    dyUni = Ly / Ncy;
    for ( int i = 1; i < Npy ; i ++ )
      {
	dy[i-1] = dyUni;
	Y[i] = Y[0] - i * dy[i-1]; 
      }
        
  }

/*
  if ( my_rank == 0 ) {
    ob.singleArrayTable(Y,Npy,1,"YtableWebbL1NonUniform");
    ob.singleArrayTable(dy,Ncy,1,"dytableWebbL1NonUniform");
  }
 */ 
 
  // z direction

  if ( ratioBCz != 1 ){
    dzcenter = MeshSize(Nczhalf,Lzhalf,ratioBCz);
    for ( int i = 0; i < Nczhalf; i++ ) {
      dz[i] = dzcenter * pow( ratioBCz, Nczhalf-1-i );
      dz[Ncz-i-1] = dz[i];
    }
    
    for ( int i = 1; i < Nczhalf; i ++ )
      {
	Z[i] = Z[i-1] - dz[i-1];
	Z[Ncz-i] = Z[Npz-i] + dz[i-1];
      }    
  }
  else if ( ratioBCz == 1 ){
    dzUni = Lz / Ncz;
    for ( int i = 1; i < Npz ; i ++ )
      {
	dz[i-1] = dzUni;
	Z[i] = Z[0] - i * dz[i-1]; 
      }    
    
  }


  if ( my_rank == 0 ) {
    ob.singleArrayTable(Z,Npz,1,"ZtableFig19-17202020Lc01uniform");
    ob.singleArrayTable(dz,Ncz,1,"dztableFig19-17202020Lc01uniform");
  }
  

   // only main processor generates tables and then arrays,
   // other processor will all copy data from the main arrays
   // these have to be defined globally.

   // to generate tables is for easy detecting errors
   // later on, will generate arrays directly on each processor
  
  X[Ncxhalf] = 0;
  Y[Ncyhalf] = 0;
  Z[Nczhalf] = 0;   


  // x direction first, then y direction
  for ( int j = 0; j < Ncy; j ++ )  
    for ( int i = 0; i < Ncx; i ++ )  
      ElementAreaTB[ j*Ncx + i ] = dx[i] * dy[j];


  // x direction first, then z direction
  for ( int j = 0; j < Ncz; j ++ )  
    for ( int i = 0; i < Ncx; i ++ )  
      ElementAreaFB[ j*Ncx + i ] = dx[i] * dz[j];
  

  // y direction first, then z direction
  for ( int j = 0; j < Ncz; j ++ )  
    for ( int i = 0; i < Ncy; i ++ )  
      ElementAreaLR[ j*Ncy + i ] = dy[i] * dz[j];


  // element volume
  // x direction, then y direction then z direction, so x-y layer by x-y layer
  for ( int i = 0; i < Ncz; i ++ )
    for ( int j = 0; j < Ncy; j ++ )
      for ( int k = 0; k < Ncx; k ++ )
	ElementVol[ i*TopBottomNo + j*Ncx + k ] = dz[i] * dy[j] * dx[k];


  
  // setting rayNo for surfaces and volume depends on area and volume size
  // local_rayNoVol and local_rayNoSurface are for minArea and minVol
  
  if ( ratioBCz <= 1 ) // boundary has the finer mesh, so minArea are at the boundaries
    {
      minAreaTB = dx[0] * dy[0];
      minAreaFB = dx[0] * dz[0];
      minAreaLR = dy[0] * dz[0];
      minArea = min(minAreaTB, minAreaFB);
      minArea = min(minArea, minAreaLR);

      minVol = dx[0] * dy[0] * dz[0];

    }
  else if ( ratioBCz >=1 ) // center has the finer mesh, so minArea are at the center
    {
      minAreaTB = dx[Ncxhalf-1] * dy[Ncyhalf-1];
      minAreaFB = dx[Ncxhalf-1] * dz[Nczhalf-1];
      minAreaLR = dy[Ncyhalf-1] * dz[Nczhalf-1];
      minArea = min(minAreaTB, minAreaFB);
      minArea = min(minArea, minAreaLR);

      minVol = dx[Ncxhalf-1] * dy[Ncyhalf-1] * dz[Nczhalf-1];

    }







  if ( my_rank == 0 ) {

   // only have rank == 0 processor generates the table,
   // otherwise, all other processors will generate the same name's tables.   
     //VolTable:
     // zup, zlow, yup, ylow, xlow, xup, vIndex, top, bottom, front, back, l, r   
     ob.VolTableMake(X, Y, Z,
		     Ncx, Ncy, Ncz,
		     TopStartNo,  BottomStartNo,
		     FrontStartNo, BackStartNo,
		     LeftStartNo,  RightStartNo,
		     TopBottomNo,  FrontBackNo, LeftRightNo,
		     "VolTableFig19-17202020Lc01uniform");  
     ToArray(VolTableSize, VolTableArray, "VolTableFig19-17202020Lc01uniform");
     
     ob.VolNeighbor(TopBottomNo, Ncx, Ncy, Ncz,
		    "VolNeighborTableFig19-17202020Lc01uniform");
     ToArray(VolNeighborSize, VolNeighborArray, "VolNeighborTableFig19-17202020Lc01uniform" );

  }



  MPI_Bcast(VolTableArray, VolTableSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(VolNeighborArray, VolNeighborSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

       
  
  MPI_Barrier (MPI_COMM_WORLD);
  
  


  double pi = 3.1415926;
  double SB = 5.669 * pow(10., -8);
  
  RNG rng;

  
  // if NoMedia == 1, then no participating media; otherwise there is participating-media
  // even if there is no participating media, still force the ray to go cell by cell
  // so that to see if this algorithm goes well
 

  double OutIntenVol, OutIntenSur, IntenFrac, LeftIntenFrac, pathlen;
  vector<double> PathLeft;
  vector<int> PathIndex;
  vector<double> PathSurfaceLeft;
  int vectorIndex;
  int offsetSurIndex;
  double previousSum, currentSum;
  double SurLeft;
  
  IntenFrac = 1e-10; //1e-20; // the percentage of Intensity left
  
  srand48 ( time ( NULL )); // for drand48()

  EnergyAmount = 1e-8; // [=] W
  

  ray obRay(BottomStartNo, FrontStartNo,
	    BackStartNo, LeftStartNo, RightStartNo,
	    sumElementNo, totalElementNo, VolElementNo,
	    LeftRightNo, Lx, Ly, Lz);
     
  int wgkaSize, coluwgka;
  coluwgka = 128; //int ( wgkaSize/4);
  wgkaSize = coluwgka * 4;

  double Tref, ucold, uhot;
  // Webb non-homogeneous  case
    double *wgkahot = new double[wgkaSize];
    double *wgkacold = new double[wgkaSize];
  
  // wgka is for CO2 N2 mixture, on the book page 624
  //  double *wgka = new double[wgkaSize];  


   // for CO2, H2O , non-homogeneous gas mixture ( Modest paper)     
    if ( INHOMOGENEOUS ) {
      
      if ( DFSK == 1 ) {
	if ( Thot  == 1000 ) 
	  ToArray(wgkaSize, wgkahot, "Tref1000-T1000-wgka-alpha25-128g");
	else if ( Thot == 2000)
	   ToArray(wgkaSize, wgkahot, "Tref2000-T2000-wgka-alpha25-128g");
	
	ToArray(wgkaSize, wgkacold, "Tref300-T300-wgka-alpha25-128g");     
      }
      else if (FSCK == 1 ) {
       
	if ( Thot == 1000 ){ // Tref = 1000K
	  ToArray(wgkaSize, wgkahot, "Tref1000-T1000-wgka-alpha25-128g");
	  ToArray(wgkaSize, wgkacold, "Tref1000-T300-wgka-alpha25-128g");
	  //   ToArray(coluwgka, a_Wall, "Tref1000-awall-alpha25-128g");
	}
	else if ( Thot == 2000 ){ // Tref = 2000K
	  ToArray(wgkaSize, wgkahot, "Tref2000-T2000-wgka-alpha25-128g");
	  ToArray(wgkaSize, wgkacold, "Tref2000-T300-wgka-alpha25-128g");
	  //   ToArray(coluwgka, a_Wall, "Tref2000-awall-alpha25-128g");
	}
	
      }
      else if (FSSK == 1){ 	 	
	Tref = Thot;
	// FSSK
	// for T= Tref, k = k(Tref) * u(Tref) = k(Tref) * 1
	// for T != Tref, k = k(Tref) * u(T) 
	// Webb non-homogeneous , non-isothermal case
	if ( Thot == 1000) {
	  uhot = 1; // Tref = Thot = 1000k
	  if ( Lc == 0.1 )
	    ucold = 0.802412; // for Lc = 10 cm
	  else if ( Lc == 0.5 )
	    ucold = 0.699813;
	  
	  ToArray(wgkaSize, wgkahot, "Tref1000-T1000-wgka-alpha25-128g");
	  ToArray(wgkaSize, wgkacold, "Tref300-T300-wgka-alpha25-128g");
	  //	ToArray(coluwgka, a_Wall, "Tref1000-awall-alpha25-128g"); //  this doesnot matter    
	}
	else if ( Thot == 2000 ) { // Thot = 2000 	  
	  uhot = 1;
	  if ( Lc == 0.1 )
	    ucold = 0.740482;
	  else if ( Lc == 0.5 )
	    ucold = 0.62502;
	  
	  ToArray(wgkaSize, wgkahot, "Tref2000-T2000-wgka-alpha25-128g");
	  ToArray(wgkaSize, wgkacold, "Tref300-T300-wgka-alpha25-128g");
	  //   ToArray(coluwgka, a_Wall, "Tref2000-awall-alpha25-128g"); //  this doesnot matter 
	}

	
      }
      
    }




   
  // ================ settting up the workingVol information,==================
  // so the code can flexiblely handle only the part interested.
  
  int local_surfaceElementNo, local_VolElementNo;
  int workingVolNo, workingSurfNo;

  workingVolNo = Nczhalf * 4; // only half of the center z column, and each layer has 4 elements

  local_VolElementNo = workingVolNo / np; // only half of the center z

  workingSurfNo = Ncx; // for back surface, two half middle column along x direction.
  local_surfaceElementNo = workingSurfNo / np;
  
  // store indices of  working control volumes  
  int *workingVol = new int[workingVolNo];

  // store indices of working surface elements
  int *workingSurf = new int[workingSurfNo];
  
   // initial as all volume elements ray no zeros first.  
   for ( int i = 0; i < VolElementNo; i ++ )
     rayNo[i] = 0;

   // initial all surface elements ray no = 0
   for ( int i = VolElementNo; i < totalElementNo; i ++ )
     rayNo[i] = 0;

   // bottom surface ray no = 1000
   for ( int i = BottomStartNo; i < FrontStartNo; i ++ )
     rayNo[i] = rayNoSurface;


   // setting up working Vol elements
  int startSliceVol; 
  startSliceVol =  ( Ncyhalf - 1 ) * Ncx  + (Ncxhalf-1);
      
  int RayIndex, rayNoi, surfaceStartNo, StartRayNoi;
  RayIndex = 0;
  
  for ( int i  = 0; i < Nczhalf; i ++ ){
    
    rayNoi = i * TopBottomNo + startSliceVol;      
    workingVol[RayIndex++] = rayNoi;
    workingVol[RayIndex++] = rayNoi+1;
    
    rayNoi = rayNoi + Ncx;
    workingVol[RayIndex++] = rayNoi;
    workingVol[RayIndex++] = rayNoi + 1;
	 
  }


  // set up working surface elements for bottom surfaces two half middle columns
  RayIndex = 0;
  StartRayNoi = BottomStartNo + ( Ncyhalf -1 ) * Ncx;
  surfaceStartNo = rayNoi;
  
  for ( int i = 0; i < Ncxhalf; i ++ ){
    rayNoi = StartRayNoi + i;
    workingSurf[RayIndex++] = rayNoi;
    workingSurf[RayIndex++] = rayNoi + Ncx;
  }

  // ob.singleIntArrayTable(workingSurf,
  //			 Ncx,
  //			 1, "workingSurf");
  
  // =========== end of setting up info for working Vol ==================





  // set up properties for surfaces and volumes
  

     /*
   //   *********  the benchmark coupled with surfaces case ***************
   
   double xx, yy, zz; // midpoint of a control volume
   double *pk;
  
   
   for ( int i = 0; i < VolElementNo ; i ++ ) {
     
     T_Vol[i] = 64.80721904; // k
     pk = VolTableArray + i * 13;
     zz =  ( * pk + * (pk + 1) ) /2;
     yy = ( * (pk + 2) + * (pk + 3) ) / 2;
     xx = ( * (pk + 4) + * (pk + 5) ) / 2;       
     kl_Vol[i] = BenchMark_k(xx, yy, zz); // set as "1" to let absorb happens all the time
     
     // let the rayNo is a function of kl
     // 0.1 is the smallest value of kl, so smallest is local_rayNoVol 
     //rayNo[i] = int (ceil(kl_Vol[i] / 0.1 * local_rayNoVol ));
     //rayNo[i] = local_rayNoVol;
     
     scatter_Vol[i] = 0;

   //  ScatFactor[i] = kl_Vol[i] / (kl_Vol[i] + scatter_Vol[i]);
     
    // emiss_Vol[i] = 0; // actually no emiss for medium ?
     // we get the emissive energy from media by kl_Vol
   }

   

  int i = 0;
   for ( ; i < BottomStartNo - VolElementNo; i ++ ) { // top surface
     //rd_surface[i] = 0.57;
     rs_surface[i] = 0.02;
     emiss_surface[i] = 0.9;
     //rs_surface[i] =  1 - emiss_surface[i]; // totally specular ( pure mirror surfaces )	 
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface??
     //rs_surface[i] =  1 - rd_surface[i] - absorb_surface[i];
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 1200;
   }
   
   for ( ; i < FrontStartNo - VolElementNo; i ++ ) { // bottom surface
     //rd_surface[i] = 0.04;
     rs_surface[i] = 0.04;
     emiss_surface[i] = 0.8;
     //rs_surface[i] =  1 - emiss_surface[i]; // totally specular ( pure mirror surfaces ) 	 
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface??
     // rs_surface[i] =  1 - rd_surface[i] - absorb_surface[i];
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 900; 
   }

   for ( ; i < BackStartNo - VolElementNo; i ++ ) { // front surface
     //rd_surface[i] = 0.475;
     rs_surface[i] = 0.475;
     emiss_surface[i] = 0.05;
     //rs_surface[i] =  1 - emiss_surface[i]; // totally specular ( pure mirror surfaces )	 
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface??
     //rs_surface[i] =  1 - rd_surface[i] - absorb_surface[i];
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 1400;
   }
   
   for ( ; i < LeftStartNo - VolElementNo; i ++ ) { // back surface
     //rd_surface[i] = 0.76;
     rs_surface[i] = 0.19;
     emiss_surface[i] = 0.05;
     //rs_surface[i] =  1 - emiss_surface[i]; // totally specular ( pure mirror surfaces )	 
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface??
     //rs_surface[i] =  1 - rd_surface[i] - absorb_surface[i];
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 2000;
   }

   for ( ; i < RightStartNo - VolElementNo; i ++ ) { // left surface
     //rd_surface[i] = 0.04;
     rs_surface[i] = 0.76;
     emiss_surface[i] = 0.2;	 //rs_surface[i] =  1 - emiss_surface[i]; // totally specular ( pure mirror surfaces )	 
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface??
     //rs_surface[i] =  1 - rd_surface[i] - absorb_surface[i];
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 600;
   }
       
   for ( ; i < sumElementNo - VolElementNo; i ++ ) { // right surface
     //rd_surface[i] = 0.04;
     rs_surface[i] = 0.76;
     emiss_surface[i] = 0.2;
     //rs_surface[i] =  1 - emiss_surface[i]; // totally specular ( pure mirror surfaces )	 
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface??
     //rs_surface[i] =  1 - rd_surface[i] - absorb_surface[i];
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 600;
   }
       
   
   // ************************* end of this case ********************
   */
  


  /*
   // ************* parallel plates for testing on surface case (no media)******************
   

   double T, emissSur, rhos;
   
   for ( int i = 0; i < VolElementNo ; i ++ ) {
     
     T_Vol[i] = 0; // k      
     kl_Vol[i] = 0; // set as "1" to let absorb happens all the time
     a_Vol[i] = 1;
     scatter_Vol[i] = 0;
    // emiss_Vol[i] = 0; // actually no emiss for medium ?
     // we get the emissive energy from media by kl_Vol
   }


   // emiss*SB*T^4 =1

   if (casePlates == 11){
     emissSur = 0.9;
     rhos = 0;
   }
   else if(casePlates == 12) {
     emissSur = 0.9;
     rhos = 0.1;
   }
   else if ( casePlates == 21 ) {
     emissSur = 0.5;
     rhos = 0;
   }
   else if ( casePlates == 22 ) {
     emissSur = 0.5;
     rhos = 0.25;
   }
   else if ( casePlates == 23 ) {
     emissSur = 0.5;
     rhos = 0.5;
   }
   else if ( casePlates == 31 ) {
     emissSur = 0.1;
     rhos = 0;
   }
   else if ( casePlates == 32 ) {
     emissSur = 0.1;
     rhos = 0.6;
   }
   else if ( casePlates == 33 ) {
     emissSur = 0.1;
     rhos = 0.9;
   }
   
  
   T = sqrt(sqrt(1/emissSur/SB));

   // top bottom surfaces has the property
   // front and back surfaces are pure specular cold surfaces
   // left and right surfaces are cold black surfaces

   int i = 0;
   for ( ; i < BottomStartNo - VolElementNo; i ++ ) { // top surface
     rs_surface[i] = rhos;
     emiss_surface[i] = emissSur;
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = T;
   }
     
   for ( ; i < FrontStartNo - VolElementNo; i ++ ) { // bottom surface
     rs_surface[i] = rhos;
     emiss_surface[i] = emissSur;
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = T;
   }
     
   for ( ; i < BackStartNo - VolElementNo; i ++ ) { // front surface--mirror
     rs_surface[i] = 1;
     emiss_surface[i] = 0;
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 0;
   }
     
   for ( ; i < LeftStartNo - VolElementNo; i ++ ) { // back surface--mirror
     rs_surface[i] = 1;
     emiss_surface[i] = 0;
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface
     rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 0;
   }
     
   for ( ; i < RightStartNo - VolElementNo; i ++ ) { // left surface--black
     rs_surface[i] = 0;
     emiss_surface[i] = 1;
     absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface
     rd_surface[i] = 1 - rs_surface[i] - absorb_surface[i];
     T_surface[i] = 0;
   }
     
     for ( ; i < sumElementNo - VolElementNo; i ++ ) { // right surface--black
       rs_surface[i] = 0;
       emiss_surface[i] = 1;
       absorb_surface[i] = emiss_surface[i]; // for gray diffuse surface
       rd_surface[i] =  1 - rs_surface[i] - absorb_surface[i];
       T_surface[i] = 0;
     }     
  
   // ********* end of parallel plates case ******************************
   
  */






  
  /*
   //  ************************ benchmark case **********************
   
  
   // benchmark case
   // with participating media, and all cold black surfaces around
  
  double xx, yy, zz; // midpoint of a control volume
  double *pk;
  
  
  // will plug in T function with position
  
  for ( int i = 0; i < VolElementNo ; i ++ ) {
    
    T_Vol[i] = 64.80721904; // k
    pk = VolTableArray + i * 13;
    zz =  ( * pk + * (pk + 1) ) /2;
    yy = ( * (pk + 2) + * (pk + 3) ) / 2;
    xx = ( * (pk + 4) + * (pk + 5) ) / 2;
    
    kl_Vol[i] = BenchMark_k(xx, yy, zz); // set as "1" to let absorb happens all the time
    a_Vol[i] = 1;
    scatter_Vol[i] = 0;
    
  }
  
  
  for ( int i = 0; i < surfaceElementNo; i ++ ) {
    rs_surface[i] = 0;
    rd_surface[i] = 0;
    absorb_surface[i] = 1 - rs_surface[i] - rd_surface[i];
    emiss_surface[i] = absorb_surface[i]; // for gray diffuse surface??
  }
  
  for ( int i = 0; i < TopBottomNo; i ++ ) // top surface
    T_surface[i] = 0;
  
  for ( int i = TopBottomNo; i < 2 * TopBottomNo; i ++ ) // bottom surface
    T_surface[i] = 0;
  
  //  set side walls
  for ( int i = FrontStartNo - TopStartNo; i < surfaceElementNo; i ++ ) // side surface
    T_surface[i] = 0;
  
  
  // *************************** end of benchmark case **********************   
  */
   



     /*   
// ====== Case 6 in Chapter 3 =====
     // cout << " i am here , line 766 - just before initial values" << endl;
   
   double *CO2 = new double [VolElementNo];
   double *H2O = new double [VolElementNo];
   double *SFV = new double [VolElementNo];
   // and we already have T_Vol for control volumes.

   int i_index;
   double xaveg;
   // as the properties only change with x, so calculate x's first
   // then simply assign these values to ys and zs.
   
   for ( int i = 0; i < xno; i ++ ) {
     
     xaveg = ( X[i] + X[i+1] + xc )/2;
     CO2[i] = 0.4 * xaveg * ( 1 - xaveg ) + 0.06;
     H2O[i] = 2 * CO2[i];
     SFV[i] = ( 40 * xaveg * ( 1 - xaveg) + 6 ) * 1e-7;
     T_Vol[i] = 4000 * xaveg * ( 1 - xaveg ) + 800;

     // for all ys and zs
     for ( int m =  0; m < zno; m ++ ) {
       for ( int n = 0; n < yno; n ++ ) {
	 i_index = i + xno * n + TopBottomNo * m;
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

   
   for ( int i = 0; i < VolElementNo; i ++ )
     scatter_Vol[i] = 0;


   obRadCoeff.PrepCoeff(CO2, H2O, SFV, T_Vol, kl_Vol,
			VolElementNo, TopBottomNo,
			xno, yno, zno);
   

   cout << " here line 816, done with propreties " << "rank = " << my_rank << endl;

   // all cold black surfaces
//    for ( int i = 0; i < surfaceElementNo; i ++ ) {
//      rs_surface[i] = 0;
//      rd_surface[i] = 0;
//      absorb_surface[i] = 1; 
//      emiss_surface[i] = 1; 
//    }


   
   // making the font, back, left and right surfaces as mirrors
   // so the top and bottom surfaces would be infinite big.
   
   int im = 0;
   for ( ; im < BottomStartNo - VolElementNo; im ++ ) { // black cold top surface
     rs_surface[im] = 0;
     emiss_surface[im] = 1;
     absorb_surface[im] = emiss_surface[im]; 
     rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }
   
     
   for ( ; im < FrontStartNo - VolElementNo; im ++ ) { // black cold bottom surface
     rs_surface[im] = 0;
     emiss_surface[im] = 1;
     absorb_surface[im] = emiss_surface[im]; 
     rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }

   
   for ( ; im < BackStartNo - VolElementNo; im ++ ) { // front surface mirror
     rs_surface[im] = 1;
     emiss_surface[im] = 0;
     absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
     rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }

   
   for ( ; im < LeftStartNo - VolElementNo; im ++ ) { // back surface mirror
     rs_surface[im] = 1;
     emiss_surface[im] = 0;
     absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
     rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }


   for ( ; im < RightStartNo - VolElementNo; im ++ ) { // left surface mirror
     rs_surface[im] = 1;
     emiss_surface[im] = 0;
     absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
     rd_surface[im] = 1 - rs_surface[im] - absorb_surface[im];
   }
     
     for ( ; im < sumElementNo - VolElementNo; im ++ ) { // right surface mirror
     rs_surface[im] = 1;
     emiss_surface[im] = 0;
     absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
     rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
     }     
  

     
   for ( int i = 0; i < TopBottomNo; i ++ ) // top surface
     T_surface[i] = 0;
   
   for ( int i = TopBottomNo; i < 2 * TopBottomNo; i ++ ) // bottom surface
     T_surface[i] = 0;
   
   //  set side walls
   for ( int i = FrontStartNo - TopStartNo; i < surfaceElementNo; i ++ ) // side surface
     T_surface[i] = 0;

   
   // ============= end of Case 6 in chapter 3 ===
   */
  


   
  // top and bottom are the parallel plates at emiss = 1, T = 0K
  // all side surfaces are mirrors.
   
   int im = 0;
   for ( ; im < BottomStartNo - VolElementNo; im ++ ) { // top surface

       rs_surface[im] = 0;
       emiss_surface[im] = 1;
       absorb_surface[im] = emiss_surface[im]; 
       rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }
   
     
   for ( ; im < FrontStartNo - VolElementNo; im ++ ) { // bottom surface

       rs_surface[im] = 0;
       emiss_surface[im] = 1;
       absorb_surface[im] = emiss_surface[im]; 
       rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }

   
   for ( ; im < BackStartNo - VolElementNo; im ++ ) { // front surface mirror 

       rs_surface[im] = 1;
       emiss_surface[im] = 0;
       absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
       rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }

   
   for ( ; im < LeftStartNo - VolElementNo; im ++ ) { // back surface mirror 

       rs_surface[im] = 1;
       emiss_surface[im] = 0;
       absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
       rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
   }


   for ( ; im < RightStartNo - VolElementNo; im ++ ) { // left surface mirror

       rs_surface[im] = 1;
       emiss_surface[im] = 0;
       absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
       rd_surface[im] = 1 - rs_surface[im] - absorb_surface[im];
   }
     
     for ( ; im < sumElementNo - VolElementNo; im ++ ) { // right surface mirror 

       rs_surface[im] = 1;
       emiss_surface[im] = 0;
       absorb_surface[im] = emiss_surface[im]; // for gray diffuse surface
       rd_surface[im] =  1 - rs_surface[im] - absorb_surface[im];
     }     
  
   
   
    for ( int i = 0; i < TopBottomNo; i ++ ) // top surface
      T_surface[i] = 0;
   
    for ( int i = TopBottomNo; i < 2 * TopBottomNo; i ++ ) // bottom surface
      T_surface[i] = 0;
   

    for ( int i = FrontStartNo - TopStartNo; i < surfaceElementNo; i ++ ) // side surface
      T_surface[i] = 0;
   



   // Webb case , 3 layers in z direction; with top and bottom at 1000K black.
   int zI, zII; // indices for z
   int zi;
   
   zi = 1;
   double firstlayer = Lzhalf - Lh;
   //   cout << "firstLayer = " << firstlayer << endl;
   
   if ( INHOMOGENEOUS ) {

     // Webb case , 3 layers in z direction; with top and bottom at 1000K e=0.8.
     
     do {
       zi++;
     }while( Z[zi] > firstlayer );
     
     zI = zi;
     //     cout << " zI = " << zI << endl;
     // note: dont need to do zi -1 for finding the firstlayer index  zI
     

    
     // cout << " zII = " << zII << endl;
     // note : need to do the zi - 1 for finding the secondlayer index zII

     // you will realize that when u take a look at the Ztable :)
     
     // first layer
     int Ti;
     for ( Ti = 0; Ti < zI * Ncx * Ncy; Ti ++ ){
       T_Vol[Ti] = Thot;
     }

     // second layer
     for ( ; Ti < VolElementNo; Ti ++ ){
       T_Vol[Ti] = Tcold;
     }

    
     // no scattering
     for ( int i = 0; i < VolElementNo; i++ )
       scatter_Vol[i] = 0;
     
           
   }



   //   if ( my_rank == 0 )
   //  ob.singleArrayTable(T_Vol, VolElementNo, 1,"TVolTable");   

    
// store all intensity for all elements
// get intensity for all elements in g-space at the same gg
  double *IntenArray = new double [totalElementNo];
  int offset_index;
     
  double theta, phi;
  double random1, random2;
  double s[3];
  
  int vIndex; // get vol index from RealSurface's derived surface classes
  int Total;
  double sumQsurface = 0;
  double sumQvolume = 0;
  
  double _alow, _aup, _blow, _bup, _constv;
  // ray's emitting surface element's index, and hitting surface element's index
  int emissSurfaceIndex, hitSurfaceIndex;
  int ii;


  // when ProcessorDone == 0, fresh start on a surface
  // when ProcessorDone == 1 , it is done with this processor calculation.
  // when ProcessorDone == 2, continue calculating onto another surface
  
  int local_Counter, ProcessorDone;  
  int my_rank_Start_No;
 
  // define these pointer arrays out of any if or loop,
  // otherwise they only exist within that domain instead of whole program
    
  RealSurface *RealPointer;
  TopRealSurface *obTop = new TopRealSurface[TopBottomNo];
  BottomRealSurface *obBottom = new BottomRealSurface[TopBottomNo];
  FrontRealSurface *obFront = new FrontRealSurface[FrontBackNo];
  BackRealSurface *obBack = new BackRealSurface[FrontBackNo];
  LeftRealSurface *obLeft = new LeftRealSurface[LeftRightNo];
  RightRealSurface *obRight = new RightRealSurface[LeftRightNo];
  
  double *netIntenSurface= new double[local_surfaceElementNo * coluwgka];
  double *netIntenVol = new double[local_VolElementNo * coluwgka];
  
  double *global_qsurface = new double [surfaceElementNo];
  double *global_Qsurface = new double [surfaceElementNo];  
  double *global_qdiv = new double [VolElementNo];
  double *global_Qdiv = new double [VolElementNo];
      
   if ( rayNoSurface != 0) { // have rays emitting from surface elements
    
    // ======  generate all real surface objects =======
    
    // has to do this seperately, top, bottom, front , back , left and right all individually.
    // because of "surfaceIndex "
    
    // *************************** Top & Bottom Surfaces ************* //
    surfaceIndex = TopStartNo;
    
    for ( int i = 0; i < TopBottomNo; i ++ ){
      obTop[i].setData(surfaceIndex, TopBottomNo);
      surfaceIndex ++;
    }
    
        
    for ( int i = 0; i < TopBottomNo; i ++ ){
      obBottom[i].setData(surfaceIndex, TopBottomNo, VolElementNo);     
      surfaceIndex ++;
    }
    
    
    // ****************** Front & Back Surfaces ****************** //
    
    for ( int i = 0; i < FrontBackNo; i ++ ) {
      obFront[i].setData(surfaceIndex, TopBottomNo, Ncx);
      surfaceIndex ++;
    }
    
    for ( int i = 0; i < FrontBackNo; i ++ ) {
      obBack[i].setData(surfaceIndex, TopBottomNo, Ncx);     
      surfaceIndex ++;
    }
    
    // *************** Left & Right Surfaces ********************** //
    
    for ( int i = 0; i < LeftRightNo; i ++ ) {
      obLeft[i].setData(surfaceIndex, Ncx);
      surfaceIndex ++;
    } 
    
    for ( int i = 0; i < LeftRightNo; i ++ ) {
      obRight[i].setData(surfaceIndex, Ncx);     
      surfaceIndex ++;
    } 
    
 }




  
  // integrate Ig over g
  
   // for ( int iggNo = 0; iggNo < ggNo; iggNo ++ ) {
   
   for ( int iggNo = 0; iggNo < coluwgka; iggNo ++ ) { //coluwgka;  iggNo ++ ) {  //coluwgka
     
     
     cout << "my_rank = " << my_rank <<  ";iggNo = " << iggNo << endl;
     
     
     // should update the intensity every iggNo step ( every different g )
     
     if ( INHOMOGENEOUS ) { // and TablePoints != 1
       int jj;
       //  cout << "here at jj" << endl;
       if ( DFSK == 1 ) {
	 // first layer T = Thot
	 for ( jj = 0; jj < zI * Ncx * Ncy; jj ++ ){
	   kl_Vol[jj] = wgkahot[iggNo*4+2] * 100;
	   a_Vol[jj] = wgkahot[iggNo*4+3];
	 }
	 
	 // second layer T = Tcold
	 for ( ; jj < VolElementNo; jj ++ ){
	   kl_Vol[jj] =  wgkacold[iggNo*4+2] * 100;
	   a_Vol[jj] =  wgkacold[iggNo*4+3];
	 }
	 
	 // for top and bottom surfaces a_surface
	 for ( int i = 0; i < surfaceElementNo; i ++ )
	   a_surface[i] = 1;
	 
       }
       else if ( FSSK == 1 ) {
	 
	 // first layer T = Thot
	 for ( jj = 0; jj < zI * Ncx * Ncy; jj ++ ){
	   kl_Vol[jj] = wgkahot[iggNo*4+2]* uhot * 100;
	   a_Vol[jj] = wgkahot[iggNo*4+3];
	 }
	 
	 // second layer T = Tcold
	 for ( ; jj < VolElementNo; jj ++ ){
	   kl_Vol[jj] =  wgkahot[iggNo*4+2]* ucold * 100;
	   a_Vol[jj] =  wgkacold[iggNo*4+3];
	 }
	 	 
	 // for top and bottom surfaces a_surface
	 for ( int i = 0; i < surfaceElementNo; i ++ )
	   a_surface[i] = 1;	 
	 
       }
       else if ( FSCK == 1 ) {
	 
	 // first layer T = Thot
	 for ( jj = 0; jj < zI * Ncx * Ncy; jj ++ ){
	   kl_Vol[jj] = wgkahot[iggNo*4+2] * 100;
	   a_Vol[jj] = wgkahot[iggNo*4+3];
	 }
	 
	 // second layer T = Tcold
	 for ( ; jj < VolElementNo; jj ++ ){
	   kl_Vol[jj] =  wgkacold[iggNo*4+2] * 100;
	   a_Vol[jj] =  wgkacold[iggNo*4+3];
	 }
	 	 
	 // for top and bottom surfaces a_surface
	 for ( int i = 0; i < surfaceElementNo; i ++ )
	   a_surface[i] = 1;
       }

             
     } // end inhomogeneous
     
  // for Volume's  Intensity
  // for volume, use black intensity
  for ( int i = 0; i < VolElementNo; i ++ )
    IntenArray[i] = obRay.VolumeIntensityBlack(i, T_Vol, a_Vol);  

  // for Surface elements' Intensity
  // for surface, using non-black intensity
  // special treatment needed for non-gray surfaces.
  
  for ( int i = VolElementNo; i < totalElementNo; i ++ ) {
    
    offset_index = i - VolElementNo;
    IntenArray[i] = obRay.SurfaceIntensity(offset_index, emiss_surface, T_surface, a_surface);
        
  }

  

  if ( rayNoSurface != 0 && OnlySurfaceCenter == 1 ) {

    int my_rank_No;
    
    my_rank_No = my_rank * local_surfaceElementNo;
    
    local_Counter = 0;
    ProcessorDone = 0;
    int SurfaceElementCounter, rayCounter, offset_index;
    double alpha; // abosorption coeff of the emitting surface element
    
    int minNo;
    
    // local_offset_startIndex and local_offset_endIndex are the local index
    // for each face surface from 0 to TopBottomNo, FrontBackNo, LeftRightNo.    
    int local_offset_startIndex, local_offset_endIndex;
    int SurfCounter;
    
    // this will only work for one processor handle one surface element
    for ( SurfCounter = 0; SurfCounter < local_surfaceElementNo; SurfCounter ++ ) {
      
       surfaceIndex = workingSurf[my_rank_No + SurfCounter];
       // cout << "my_rank = " << my_rank << ";  surfaceIndex = " << surfaceIndex <<
       //	 "; local_surfaceElementNo = " << local_surfaceElementNo << endl;
       rayNo[surfaceIndex] = rayNoSurface;
       offset_index = surfaceIndex - VolElementNo;
       SurfaceElementCounter = surfaceIndex - BottomStartNo;
       
       alpha = absorb_surface[offset_index];
       
       RealPointer = &obBottom[SurfaceElementCounter];
       
       OutIntenSur = obRay.SurfaceIntensity(offset_index,
					    emiss_surface,
					    T_surface, a_surface);
	 
 
	 double *IncomingIntenSur = new double[rayNo[surfaceIndex]];
	 for ( int i = 0; i < rayNo[surfaceIndex]; i ++ )
	   IncomingIntenSur[i] = 0;
	 
	 for ( rayCounter = 0; rayCounter < rayNo[surfaceIndex]; rayCounter++ ) {
	   
	   LeftIntenFrac = 1;
	   previousSum = 0;
	   currentSum = 0;
	   
	   SurLeft = alpha;
	   

	   // get emitting ray's direction vector s
	   // should watch out, the s might have previous values
	   RealPointer->get_s(rng, theta, random1, phi, random2, s);	   	   
	   RealPointer->get_limits(VolTableArray, vIndex);  
	   RealPointer->get_public_limits(_alow, _aup, _blow, _bup, _constv);
	   
	   // get ray's emission position 
	   obRay.getEmissionPosition(_alow, _aup,
				     _blow, _bup,
				     _constv,
				     surfaceIndex);
	   
	   obRay.set_currentvIndex(vIndex);
	   
	   // get ray's emission direction directionVector[] from obRealSurface s[]
	   obRay.get_directionS(s);
	   
	   vectorIndex = 0;
	   
	   
	   // with participating media
	   
	   // only one criteria for now ( the left energy percentage )
	   do {
	     
	     previousSum = currentSum;
	     
	     
	     // checking scattering first
	     // if hit on virtual surface, PathSurfaceLeft is updated.
	     // else no update on PathSurfaceLeft.
	     obRay.TravelInMediumInten(kl_Vol, scatter_Vol,
				       VolNeighborArray, VolTableArray,
				       PathLeft, PathIndex, PathSurfaceLeft);
	     
	     // cout << "my_rank = " << my_rank << "at line 2092 " << endl;	     
	     // the upper bound of the segment
	     currentSum = previousSum + PathLeft[vectorIndex];
	     
	     
	     // the IntensityArray for volumes are black ones.
	     // use the previous SurLeft here.
	     // SurLeft is not updated yet.
	     
	     IncomingIntenSur[rayCounter] = IncomingIntenSur[rayCounter] + 
	       IntenArray[PathIndex[vectorIndex]]  
	       * ( exp(-previousSum) - exp(-currentSum) ) * SurLeft;
	     
	     
	     
	     hitSurfaceIndex = obRay.get_hitSurfaceIndex();
	     
	     // if hit on virtual surface, then it's already been taken care of
	     // new direction already set in obRay.TravelInMedium function
	     
	     // if hit on real surface, then energy needs to be attenuated again by surface property 
	     // and find new direction for the ray after reflection 
	     
	     if ( hitSurfaceIndex != -1 ) { // hit on real surface
	       
	       // PathSurfaceLeft is updated here
	       obRay.hitRealSurfaceInten(absorb_surface,
					 rs_surface,
					 rd_surface,
					 PathSurfaceLeft);
	    
	       // but use the previous SurLeft
	       IncomingIntenSur[rayCounter] = IncomingIntenSur[rayCounter] +
		 IntenArray[hitSurfaceIndex] * exp (-currentSum ) * SurLeft;	    
	      
	     }
	     
	     
	     // set hitPoint as new emission Point
	     // and direction of the ray already updated
	     obRay.setEmissionPosition();
	     
	     LeftIntenFrac = exp(-currentSum) * SurLeft;
	     
	     
	     // this way, SurLeft is updated no matter real surface or virtual.
	     // SurLeft is updated for the next step's calculation
	     
	     SurLeft = SurLeft *PathSurfaceLeft[vectorIndex];
	     vectorIndex ++;
	     
	     
	     
	   }while ( LeftIntenFrac >= IntenFrac);
	   
	   
	   PathLeft.clear();
	   PathIndex.clear();
	   PathSurfaceLeft.clear();
	   
	 } // rayCounter loop
	 

       
	 sumIncomInten = 0;
	 for ( int aaa = 0; aaa < rayNo[surfaceIndex]; aaa ++ )
	   sumIncomInten = sumIncomInten + IncomingIntenSur[aaa];
	 
	 delete[] IncomingIntenSur;
	 
	 aveIncomInten = sumIncomInten / rayNo[surfaceIndex];
	 //	 cout << " I am here in the bottom loop1" << endl;
	 netIntenSurface[ iggNo * local_surfaceElementNo + local_Counter ] =
	   OutIntenSur - aveIncomInten;

	 // cout << " I am here in the bottom loop" << endl;
	 local_Counter ++;

       
       // ======================= end of BottomSurface ===============  
       
       
       
    } // end if 
    
  }
  
  //  cout << " i am here after one iggNo" << endl;
  
  
   }// end gg loop

   MPI_Barrier (MPI_COMM_WORLD);
   
   

  // surface cell
  
  double *integrIntenSurface = new double[local_surfaceElementNo];
  double *global_IntenSurface = new double[surfaceElementNo];
  int *displs = new int[np];
  int *rcounts = new int[np];
  
  if ( rayNoSurface != 0 ) {
    
    // initialize integrIntenSurface;
    for ( int i = 0; i < local_surfaceElementNo; i ++ )
      integrIntenSurface[i] = 0;
    
    MPI_Barrier (MPI_COMM_WORLD);

     
     // using Gausian quadrature integration
 
     for ( int i = 0; i < local_surfaceElementNo; i ++ ){
     
        for ( int j = 0; j < coluwgka; j ++ ) {
	  integrIntenSurface[i] = integrIntenSurface[i] + //netIntenSurface[j*local_surfaceElementNo + i];
	 wgkahot[j*4] * netIntenSurface[j*local_surfaceElementNo + i];	
	 }
     }
    
    
    

       
     
     
     if ( my_rank == 0 ) {

       for ( int i = 0; i < surfaceElementNo; i ++ )
	   global_IntenSurface[i] = 0;

     }


    for (int i = 0; i < np; ++i)
      {
	displs[i] = i * local_surfaceElementNo;
	rcounts[i] = local_surfaceElementNo;
      }

     
    // then merge all processors surface elements
    
    MPI_Barrier (MPI_COMM_WORLD);
    
    MPI_Gatherv(integrIntenSurface, local_surfaceElementNo, MPI_DOUBLE,
 		global_IntenSurface, rcounts, displs, MPI_DOUBLE,
 		0, MPI_COMM_WORLD);
    
    MPI_Barrier (MPI_COMM_WORLD);
    

    if ( my_rank == 0 ) {


      int offset_index;

      // top surfaces
      for ( offset_index = 0; offset_index < (BottomStartNo-VolElementNo); offset_index ++ ){
	global_qsurface[offset_index] = pi * global_IntenSurface[offset_index];
	global_Qsurface[offset_index] = global_qsurface[offset_index] *
	  ElementAreaTB[offset_index];
	
      }


      // bottom surfaces
      for ( ; offset_index < (FrontStartNo-VolElementNo); offset_index ++ ){
	global_qsurface[offset_index] = pi * global_IntenSurface[offset_index];
	global_Qsurface[offset_index] = global_qsurface[offset_index] *
	  ElementAreaTB[offset_index - TopBottomNo];
	
      }

      
      // front surfaces
      for ( ; offset_index < (BackStartNo-VolElementNo); offset_index ++ ) {	
	global_qsurface[offset_index] = pi * global_IntenSurface[offset_index];	
	global_Qsurface[offset_index] = global_qsurface[offset_index] *
	  ElementAreaFB[offset_index - 2 * TopBottomNo];
	
      }
      

      // back surfaces
      for ( ; offset_index < (LeftStartNo-VolElementNo); offset_index ++ ) {	
	global_qsurface[offset_index] = pi * global_IntenSurface[offset_index];	
	global_Qsurface[offset_index] = global_qsurface[offset_index] *
	  ElementAreaFB[offset_index - 2 * TopBottomNo - FrontBackNo];
	
      }

      
      // left surfaces
      for ( ; offset_index < (RightStartNo-VolElementNo); offset_index ++ ) {	
	global_qsurface[offset_index] = pi * global_IntenSurface[offset_index];	
	global_Qsurface[offset_index] = global_qsurface[offset_index] *
	  ElementAreaLR[offset_index - 2 * TopBottomNo - 2 * FrontBackNo];
	
      }
      

      // right surfaces
      for ( ; offset_index < surfaceElementNo; offset_index ++ ) {	
	global_qsurface[offset_index] = pi * global_IntenSurface[offset_index];	
	global_Qsurface[offset_index] = global_qsurface[offset_index] *
	  ElementAreaLR[offset_index -  2 * TopBottomNo - 2 * FrontBackNo - LeftRightNo];
	
      }

      

      ob.vtkSurfaceTableMake("vtkSurfaceFSCKFig19-17-Lc01Tref2000K-128g-202020uniform", Npx, Npy, Npz,
			     X, Y, Z, surfaceElementNo,
			     global_qsurface, global_Qsurface);
       
      
      for ( int i = 0; i < surfaceElementNo; i ++ )
	sumQsurface = sumQsurface + global_Qsurface[i] ;
      
      
    } // end if my_rank == 0
    
    
    
  } // end if local_rayNoSurface != 0 
  


  MPI_Barrier (MPI_COMM_WORLD);




  
  // Vol cell
  
  double *integrIntenVol = new double[local_VolElementNo];
  double *global_IntenVol = new double[VolElementNo];
  
  if ( rayNoVol != 0 ) {

    for ( int i = 0; i < local_VolElementNo; i ++ )
      integrIntenVol[i] = 0;
    
    MPI_Barrier (MPI_COMM_WORLD);

    
    // wgka597 and all wgka's weights w are all the same
    
     for ( int i = 0; i < local_VolElementNo; i ++ ){
       for ( int j = 0; j < coluwgka; j ++ ) { // coluwgka
	 integrIntenVol[i] = integrIntenVol[i] + // netIntenVol[j*local_VolElementNo+i];
	  wgkahot[j*4] * netIntenVol[j*local_VolElementNo+i];
      }
     }
    

    
     
  
     if ( my_rank == 0 ) {

       for ( int i = 0; i < VolElementNo; i ++ )
	   global_IntenVol[i] = 0;

     }


    for (int i = 0; i < np; ++i)
      {
	displs[i] = i * local_VolElementNo;
	rcounts[i] = local_VolElementNo;
      }

     
    // then merge all processors surface elements
    MPI_Barrier (MPI_COMM_WORLD);
            
    MPI_Gatherv(integrIntenVol, local_VolElementNo, MPI_DOUBLE,
		global_IntenVol, rcounts, displs, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
        
    MPI_Barrier (MPI_COMM_WORLD);

 
    if ( my_rank == 0 ) {
      
      int offset_index = 0;
      
      for ( ; offset_index < VolElementNo; offset_index ++ ) {
	
	global_qdiv[offset_index] = 4 * pi * global_IntenVol[offset_index];
	global_Qdiv[offset_index] = global_qdiv[offset_index] * ElementVol[offset_index];
	
      }
           
      
      for ( int i = 0; i < VolElementNo; i ++ )
	sumQvolume = sumQvolume + global_Qdiv[i];
      
      
      //  ob.vtkVolTableMake("vtkVolBenchmark404040uniform", Npx, Npy, Npz,
      //			 X, Y, Z, VolElementNo,
      //			 global_qdiv, global_Qdiv);
      
    }
    
    
    
  } // end if local_rayNoVol != 0



  /*
  
  if ( my_rank == 0 ){
    //  cout << "sumQsurface = " << sumQsurface << endl;
    cout << "sumQvolume = " << sumQvolume << endl;
    //  double difference;
    //  difference = sumQsurface + sumQvolume;
    
    // cout << " the heat balance difference = (sumQsurface + sumQvolume) = " <<
    //    difference << endl;
    
    //  double Frac;
    //   Frac = difference / sumQsurface;

    // cout << " Frac = " << Frac << endl;
  }
  */


  // cout << "local_surfaceElementNo = " << local_surfaceElementNo << endl;
  
  delete[] T_Vol;
  delete[] kl_Vol;   
  delete[] scatter_Vol;
  delete[] rayNo;
  delete[] T_surface;
  delete[] absorb_surface;
  delete[] rs_surface;
  delete[] rd_surface;
  delete[] emiss_surface;   
  delete[] VolTableArray;
  delete[] VolNeighborArray;
  delete[] X;
  delete[] Y;
  delete[] Z;

  delete[] dx;
  delete[] dy;
  delete[] dz;
  delete[] ElementAreaTB;
  delete[] ElementAreaFB;
  delete[] ElementAreaLR;
  delete[] ElementVol;
  
  delete[] obTop;
  delete[] obBottom;
  delete[] obFront;
  delete[] obBack;
  delete[] obLeft;
  delete[] obRight;
  
  //delete[] CO2;
   // delete[] H2O;
  // delete[] SFV;
  
  //  delete[] kgp;

  
  //  delete[] kk_g;
  // delete[] Rg;
  // delete[] gg;
  
  //   delete[] kgpzone2;
  //   delete[] kgpzone3;
  //   delete[] kgpzone4;
  
  // delete[] kgVoltest;
  // delete[] kgaVoltest;
  // delete[] a_g;
  
  delete[] netIntenSurface;
  delete[] netIntenVol;
  delete[] global_IntenSurface;
  delete[] global_IntenVol;
  
  
  
  //  if ( ggNo != 1 ) {
    delete[] integrIntenSurface;
    delete[] integrIntenVol;
    //  }
  
    
//   delete[] wgka597;
//   delete[] wgka777;
//   delete[] wgka937;
//   delete[] wgka1077;
//   delete[] wgka1197;
//   delete[] wgka1297;
//   delete[] wgka1377;
//   delete[] wgka1437;
//   delete[] wgka1477;
//   delete[] wgka1497;

   delete[] wgkahot;
   delete[] wgkacold;
  
  delete[] global_qdiv;
  delete[] global_Qdiv;
  delete[] global_qsurface;
  delete[] global_Qsurface;
  delete[] a_surface;
  
  delete[] displs;
  delete[] rcounts;
  delete[] workingVol;
  delete[] workingSurf;
  //  delete[] wgka;
  
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  time2 = MPI_Wtime();
  
  if ( my_rank == 0 ) {
	cout << " Lx = " << Lx << "; Ly = " << Ly << " ; Lz = " << Lz << endl;
	cout << " Ncx = " << Ncx << " ; Ncy = " << Ncy << "; Ncz = " << Ncz << endl;
	cout << " ratioBCx = " << ratioBCx << "; ratioBCy = " << ratioBCy << "; ratioBCz = " << ratioBCz << endl;
    cout << "processor used np = " << np << endl;
    cout << " time used up (S) = " << time2 - time1;
    cout << "sec with a precision of " << precision;
    cout << "sec." << endl;
  }

  
  MPI_Finalize();

  
  return 0;


}

