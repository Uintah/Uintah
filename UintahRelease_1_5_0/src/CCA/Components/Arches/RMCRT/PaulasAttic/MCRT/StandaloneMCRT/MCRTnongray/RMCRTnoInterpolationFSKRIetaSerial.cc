/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
// non gray gas solver. using WSGG, RadCoeff, and FSK.
//------- BackwardMCRTSolver.cc-----
// ------ Backward (Reverse ) Monte Carlo Ray-Tracing Radiation Model------
#include "Surface.h"
#include "RealSurface.h"
#include "TopRealSurface.h"
#include "BottomRealSurface.h"
#include "FrontRealSurface.h"
#include "BackRealSurface.h"
#include "LeftRealSurface.h"
#include "RightRealSurface.h"
#include "VirtualSurface.h"
#include "ray.h"
#include "VolElement.h"
#include "MakeTableFunction.h"
#include "MersenneTwister.h"
#include "Consts.h"
#include "RadCoeff.h"
#include "RadWsgg.h"
#include "BinarySearchTree.h"
//#include "mpi.h"

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


// unlike RMCRTnoInterpolationFSK.cc or RMCRTnoInterpolationNongray.cc
// in this case, there is no calculation over iggNo times
// just do the calculation once, but with many more rays,
// to be able to cover all g.
  
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

// to be able to do integration int_0^1 I_g dg,
// it is convenient that the g picked up every time is in accending order
// because R--g--k are all in accending order
// and R is supposed to be a uniformly generated random number from 0 to 1.
// lets set R in accending order, thus g and k will all be corresponding in accending order

template<class SurfaceType>
void rayfromSurf(SurfaceType &obSurface,
		 RealSurface *RealPointer,
		 VirtualSurface &obVirtual,
		 ray &obRay,
		 MTRand &MTrng,
		 const int &surfaceFlag,
		 const int &surfaceIndex,
		 const double * const alpha_surface[],
		 const double * const emiss_surface[],
		 const double * const T_surface[],
		 const double * const a_surface[],
		 const double * const rs_surface[],
		 const double * const rd_surface[],
		 double *IntenArray_Vol,
		 const double * const IntenArray_surface[],
		 const double *X, const double *Y, const double *Z,
		 double *kl_Vol, const double *scatter_Vol,
		 const int *VolFeature,
		 const int &thisRayNo,
		 const int &iIndex,
		 const int &jIndex,
		 const int &kIndex,
		 const int &iggNo,
		 const double &StopLowerBound,
		 double ***netInten_surface,
		 double *s,
		 BinarySearchTree &obBST,
		 const int &gSize, const int &VolElementNo,
		 const double *Rkg, const double *gk,
		 const double *kl, const double *g, const double *Ibeta){
  
  double alpha, previousSum, currentSum, LeftIntenFrac, SurLeft;
  double PathLeft, PathSurfaceLeft, weight, traceProbability;
  double OutIntenSur, sumIncomInten, aveIncomInten;
  int rayCounter, hitSurfaceFlag, hitSurfaceIndex;
  double Rgg;
  
  // get surface element's absorption coefficient
  alpha = alpha_surface[surfaceFlag][surfaceIndex];
  OutIntenSur = IntenArray_surface[surfaceFlag][surfaceIndex];
   
  double *IncomingIntenSur = new double[ thisRayNo ];
  
	  // loop over ray numbers on each surface element
  for ( rayCounter = 0; rayCounter < thisRayNo; rayCounter++ ) {

    /*
    Rgg =  MTrng.randExc(); // Rgg random number
    // cout << " Rgg = " << Rgg << endl;
    obBST.search(Rgg, Rkg, gSize);
    obBST.calculate_gk(gk, Rkg, Rgg);
    g = obBST.get_g();
    //  cout << " g= "<< g << endl;
    kl = obBST.get_k();
    // cout << "kl = " << kl << endl;
    countg[obBST.get_lowI()] ++;

    */


    // set absorption coeff
   for ( int ki = 0; ki < VolElementNo; ki++){
     IntenArray_Vol[ki] = Ibeta[rayCounter];
      kl_Vol[ki] = kl[rayCounter];
   }
   
    LeftIntenFrac = 1;
    traceProbability = 1;
    weight = 1;
    previousSum = 0;
    currentSum = 0;
    IncomingIntenSur[rayCounter] = 0;
    
    // set SurLeft = absorption coeff here is because the Intensity is
    // attenuated on the real surface by absorption.
    
    SurLeft = alpha;
    
    // get emitting ray's direction vector s
    // should watch out, the s might have previous values
    RealPointer->get_s(MTrng, s);    
    RealPointer->get_limits(X, Y, Z);
    
    
    // get ray's emission position, xemiss, yemiss, zemiss
    obRay.set_emissP(MTrng,
		     obSurface.get_xlow(), obSurface.get_xup(),
		     obSurface.get_ylow(), obSurface.get_yup(),
		     obSurface.get_zlow(), obSurface.get_zup());
    
    //   obRay.set_straightP();
    //   obRay.set_straight_len(0);
    
    obRay.set_directionS(s);
    obRay.set_currentvIndex(iIndex, jIndex, kIndex);
    obRay.dirChange = 1;
    
    do {
      
      weight = weight / traceProbability;
      previousSum = currentSum;
      
      // checking scattering first
      // if hit on virtual surface, PathSurfaceLeft is updated.
      // else no update on PathSurfaceLeft.
      obRay.TravelInMediumInten(MTrng, obVirtual,
				kl_Vol, scatter_Vol,
				X, Y, Z, VolFeature,
				PathLeft, PathSurfaceLeft);
      
      
      // the upper bound of the segment
      currentSum = previousSum + PathLeft;
      
      IncomingIntenSur[rayCounter] = IncomingIntenSur[rayCounter] + 
	IntenArray_Vol[obRay.get_currentvIndex()] 
	* ( exp(-previousSum) - exp(-currentSum) ) * SurLeft
	* weight;
      
//       cout << "previousSum = " << previousSum << endl;
//       cout << "currentSum = " << currentSum << endl;
      
      // 	      	cout << "InComing = " << IncomingIntenSur[rayCounter] << endl;
      // 	      cout << "IntenArray_Vol = " << IntenArray_Vol[obRay.get_currentvIndex()] << endl;
      
      if ( !obRay.VIRTUAL ) {
	
	hitSurfaceFlag = obRay.get_surfaceFlag();
	hitSurfaceIndex = obRay.get_hitSurfaceIndex();
// 	cout << "hitSurfaceFlag = " << hitSurfaceFlag << endl;
// 	cout << "hitSurfaceIndex = " << hitSurfaceIndex << endl;
	// PathSurfaceLeft is updated here
	// and it comes into effect for next travelling step.
	obRay.hitRealSurfaceInten(MTrng,
				  alpha_surface[hitSurfaceFlag],
				  rs_surface[hitSurfaceFlag],
				  rd_surface[hitSurfaceFlag],
				  PathSurfaceLeft);
		
	// 		cout << "obRay.get_surfaceFlag() = " << obRay.get_surfaceFlag() << endl;
	// 		cout << "surfaceFlag = " << surfaceFlag << endl;
	
	IncomingIntenSur[rayCounter] = IncomingIntenSur[rayCounter] +
	  IntenArray_surface[hitSurfaceFlag][hitSurfaceIndex] *
	  exp ( -currentSum ) * SurLeft
	  * weight;
	//	cout << "InComing = " << IncomingIntenSur[rayCounter] << endl;
      }
      
      
      // set hitPoint as new emission Point
      // and direction of the ray already updated
      obRay.update_emissP();
      obRay.update_vIndex();
      
      SurLeft = SurLeft * PathSurfaceLeft;
      
      LeftIntenFrac = exp( -currentSum) * SurLeft;
      traceProbability = min(1.0, LeftIntenFrac/StopLowerBound);
      
    }while (  MTrng.randExc() < traceProbability); // continue the path

       
  } // rayCounter loop
	  
  
  sumIncomInten = 0;
  
  for ( int aaa = 0; aaa < thisRayNo-1; aaa ++ )
    sumIncomInten = sumIncomInten +
      ( IncomingIntenSur[aaa+1] + IncomingIntenSur[aaa] ) *
         ( g[aaa+1] - g[aaa]) * 100 /2.0;
  

   delete[] IncomingIntenSur;
  
  aveIncomInten = sumIncomInten / thisRayNo;
  
  //  netInten_surface[surfaceFlag][surfaceIndex][iggNo] =
  //    OutIntenSur - aveIncomInten;

  netInten_surface[surfaceFlag][surfaceIndex][iggNo] =
    OutIntenSur - sumIncomInten;
  
  
  //	  cout << "netInten_surface = " << netInten_surface[surfaceFlag][surfaceIndex] << endl;
  
}




int main(int argc, char *argv[]){

    time_t time_start, time_end;

  
    //   int my_rank; // rank of process
    //   int np; // number of processes


  //  int casePlates;
  //  cout << " Please enter plates case " << endl;
  //  cin >> casePlates;

//   // starting up MPI
//    MPI_Init(&argc, &argv);
//    MPI_Barrier(MPI_COMM_WORLD);
  
//   precision = MPI_Wtick();
  
//   time1 = MPI_Wtime();
  
//   // Find out process rank
//    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

//   // Find out number of processes
   //  MPI_Comm_size(MPI_COMM_WORLD, &np);
  

  int rayNoSurface, rayNoVol;

  // Lhalf and dxcenter is just the distance, spacing, not the coordinate
  int Ncx, Ncy, Ncz; // number of cells in x, y, z directions
  int Npx, Npy, Npz; // number of grid points
  int Ncxhalf, Ncyhalf, Nczhalf;  // half number of the cells in cooresponding direction
  double Lx, Ly, Lz; // full length of full domain in x, y, z direction
  double Lxhalf, Lyhalf, Lzhalf; // half length of full domain in x,y,z direction
  double ratioBCx, ratioBCy, ratioBCz;  // ratioBC = Boundary cell/ center cell
  double dxcenter, dycenter, dzcenter; // center cells' size
  

  int BottomStartNo, FrontStartNo, BackStartNo, LeftStartNo, RightStartNo; 
  int VolElementNo, TopBottomNo, FrontBackNo, LeftRightNo;
  int surfaceElementNo;
  //  double EnergyAmount; // set as customer self-set-up later
  //  double sumIncomInten, aveIncomInten;  
  double StopLowerBound;
  double linear_b, eddington_f, eddington_g;
  int PhFunc;
  double scat;
  double dxconst, dyconst, dzconst; // if it is uniform mesh, dx dy dz are constant variables
  int coluwgka; // coluwgka -- size of one column of wgka array
  int gSize, gkSize, iggNo;
  
  
  BinarySearchTree obBST;
  gSize = 1495001; 
  gkSize = gSize * 2;
  iggNo = 0;

  double *Rkg = new double [gSize];
  double *gk = new double [gkSize];
   
  scat = 0.0;
  linear_b = 1;
  eddington_f = 0;
  eddington_g = 0;
  PhFunc = ISOTROPIC;
  coluwgka = 1;
  
  StopLowerBound = 1e-10;
  rayNoSurface = 0;
  rayNoVol = 1;  
  Ncx = 20;
  Ncy = 20;
  Ncz = 20;
  ratioBCx = 0.8;
  ratioBCy = 0.8;
  ratioBCz = 0.8;
  Lx = 1;
  Ly = 1;
  Lz = 1;

  if ( ratioBCx == 1 )
    dxconst = Lx/Ncx;

  if ( ratioBCy == 1 )
    dyconst = Ly/Ncy;

  if ( ratioBCz == 1 )
    dzconst = Lz/Ncz;
  
//   MPI_Barrier (MPI_COMM_WORLD);  
//   MPI_Bcast(&rayNoSurface, 1, MPI_INT, 0, MPI_COMM_WORLD);
//   MPI_Bcast(&rayNoVol, 1, MPI_INT, 0, MPI_COMM_WORLD);
//   MPI_Bcast(&Ncx, 1, MPI_INT, 0, MPI_COMM_WORLD);
//   MPI_Bcast(&Ncy, 1, MPI_INT, 0, MPI_COMM_WORLD);
//   MPI_Bcast(&Ncz, 1, MPI_INT, 0, MPI_COMM_WORLD);
//   MPI_Bcast(&ratioBCx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//   MPI_Bcast(&ratioBCy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);  
//   MPI_Bcast(&ratioBCz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//   MPI_Barrier (MPI_COMM_WORLD);


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
  surfaceElementNo = 2 * ( Ncx * Ncy + Ncx * Ncz + Ncy * Ncz );
  BottomStartNo = TopBottomNo;
  FrontStartNo = BottomStartNo + TopBottomNo;
  BackStartNo = FrontStartNo + FrontBackNo;
  LeftStartNo = BackStartNo + FrontBackNo;
  RightStartNo = LeftStartNo + LeftRightNo;
  
  int surfaceNo[6];
  surfaceNo[0] = TopBottomNo;
  surfaceNo[1] = TopBottomNo;
  surfaceNo[2] = FrontBackNo;
  surfaceNo[3] = FrontBackNo;
  surfaceNo[4] = LeftRightNo;
  surfaceNo[5] = LeftRightNo;
  
  int ghostX, ghostY, ghostZ;
  ghostX = Ncx +2;
  ghostY = Ncy +2;
  ghostZ = Ncz +2;
  
  int ghostTotalNo = ghostX * ghostY * ghostZ;
  int FLOW = 1;
  int WALL = 0;
  int *VolFeature = new int [ghostTotalNo];
  int ghosti, ghostj, ghostk;
  int ghostTB, ghostFB, ghostLR;

  ghostTB = ghostX * ghostY;
  ghostFB = ghostX * ghostZ;
  ghostLR = ghostY * ghostZ;


  // Numbering start from bottom, front left corner
  
  //  cout << "===== top ======" << endl;
  // top ghost cells
  ghostk = Ncz; // Npz - 1 
  for ( ghostj = -1; ghostj < Npy; ghostj ++ ){
    for ( ghosti = -1; ghosti < Npx; ghosti ++){
      VolFeature[(ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB]= WALL;
      // cout << (ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB << endl;
    }
    
  }


  //   cout << " ==== bottom === " << endl;
  // bottom ghost cells
  ghostk = -1;
  for ( ghostj = -1; ghostj < Npy; ghostj ++ ){
    for ( ghosti = -1; ghosti < Npx; ghosti ++){
      VolFeature[(ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB]= WALL;
      // cout << (ghosti+1) + (ghostj+1)* ghostX + (ghostk+1) * ghostTB << endl;
    }    
  }

  //   cout << " ===== front ==== " << endl;
  // front ghost cells
  ghostj = -1;
  for ( ghostk = -1; ghostk < Npz; ghostk ++ ){
    for ( ghosti = -1; ghosti < Npx; ghosti ++){
      // cout << (ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB << endl;
      VolFeature[(ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB]= WALL;
     
    }    
  }

  //  cout << " ===== back ====" << endl;
  // back ghost cells
  ghostj = Ncy;
  for ( ghostk = -1; ghostk < Npz; ghostk ++ ){
    for ( ghosti = -1; ghosti < Npx; ghosti ++){
      VolFeature[(ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB]= WALL;
      //  cout << (ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB  << endl;
    }    
  }

  // cout << "===== left ===== " << endl;
  
  // left ghost cells
  ghosti = -1;
  for ( ghostk = -1; ghostk < Npz; ghostk ++ ){
    for ( ghostj = -1; ghostj < Npy; ghostj ++){
      VolFeature[(ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB]= WALL;
      // cout << (ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB << endl;
    }    
  }


  //  cout << " ======= right ===== " << endl;
  // right ghost cells
  ghosti = Ncx; // if normal , it is Ncx - 1
  for ( ghostk = -1; ghostk < Npz; ghostk ++ ){
    for ( ghostj = -1; ghostj < Npy; ghostj ++){
      VolFeature[(ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB]= WALL;
      //  cout << (ghosti+1) + (ghostj+1) * ghostX + (ghostk+1) * ghostTB << endl;
    }    
  }

  
  for ( int k = 0; k < Ncz; k ++ )
    for ( int j = 0; j < Ncy; j ++ )
      for ( int i = 0; i < Ncx; i ++ )
	VolFeature[(i+1) + (j+1) * ghostX + (k+1) * ghostTB] = FLOW;
  
  // get coordinates arrays
  double *X = new double [Npx]; // i 
  double *Y = new double [Npy]; // j 
  double *Z = new double [Npz]; // k 
  double *dx = new double[Ncx]; // x cell's size dx
  double *dy = new double[Ncy]; // y cell's size dy
  double *dz = new double[Ncz]; // z cell's size dz

  // netInten_surface
  double ***netInten_surface; // netInten_surface(surfaceFlag, surfaceIndex, iggNo)
  netInten_surface = new double **[6]; // six faces

  // six surfaces
  // all these are equivalent to 2D matrix.
  double *alpha_surface[6], *rs_surface[6], *rd_surface[6], *IntenArray_surface[6];
  double *a_surface[6], *T_surface[6], *emiss_surface[6];
  int *rayNo_surface[6];

  for ( int i = 0; i < 6; i ++){
    alpha_surface[i] = new double [surfaceNo[i]];
    rs_surface[i] = new double [surfaceNo[i]];
    rd_surface[i] = new double [surfaceNo[i]];
    T_surface[i] = new double [surfaceNo[i]];
    emiss_surface[i] = new double [surfaceNo[i]];
    a_surface[i] = new double [surfaceNo[i]];
    rayNo_surface[i] = new int [surfaceNo[i]];
    IntenArray_surface[i] = new double [surfaceNo[i]];
    netInten_surface[i] = new double *[surfaceNo[i]];
  }

  
  // top bottom surfaces
  for ( int j = 0; j < TopBottomNo; j ++ ){
    netInten_surface[TOP][j] = new double [coluwgka];
    netInten_surface[BOTTOM][j] = new double [coluwgka];    
  }
  // front back surfaces
  for ( int j = 0; j < FrontBackNo; j ++ ){
    netInten_surface[FRONT][j] = new double [coluwgka];
    netInten_surface[BACK][j] = new double [coluwgka];    
  }
  // left right surfaces
  for ( int j = 0; j < LeftRightNo; j ++ ){
    netInten_surface[LEFT][j] = new double [coluwgka];
    netInten_surface[RIGHT][j] = new double [coluwgka];    
  }
  
   
  // get property of vol
  double *T_Vol = new double [VolElementNo];
  double *kl_Vol = new double [VolElementNo];
  double *scatter_Vol = new double [VolElementNo];
  double *a_Vol = new double[VolElementNo];
  int *rayNo_Vol = new int [VolElementNo];  
  double *IntenArray_Vol = new double [VolElementNo];
  double **netInten_Vol;
  netInten_Vol = new double *[VolElementNo];
  for ( int i = 0; i < VolElementNo; i ++)
    netInten_Vol[i] = new double[coluwgka];
  
  // ========= set values or get values for array pointers =============
  // the center of the cube is at (0,0,0) in a cartesian coordinate
  // the orgin of the cube ( domain ) can be changed here easily

  // the X, Y, Z are all face centered. so the dimensions are Npx, Npy, Npz.
   X[0] = -Lx/2.; // start from left to right
   Y[0] = -Ly/2.; // start from front to back
   Z[0] = -Lz/2; // start from bottom to top
   X[Npx-1] = Lx/2;
   Y[Npy-1] = Ly/2;
   Z[Npz-1] = Lz/2;

    // if a cell-centered variable index is i, j, k,
   // Thus the corresponding face-centered indices are:
   // face on xm is i, j, k, --- left
   // face on xp is i+1, j, k -- right 
   // face on ym is i, j, k, ---- front
   // face on yp is i, j+1, k --- back
   // face on zm is i, j, k --- bottom
   // face on zp is i, j, k+1 -- top

   int powNo;
   double dxUni, dyUni, dzUni;
   
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
  


  // y direction

  if ( ratioBCy != 1 ) {
    
    dycenter = MeshSize(Ncyhalf,Lyhalf,ratioBCy);
    for ( int i = 0; i < Ncyhalf; i++ ) {
      dy[i] = dycenter * pow( ratioBCy, Ncyhalf-1-i );
      dy[Ncy-i-1] = dy[i];
    }
    
    for ( int i = 1; i < Ncyhalf; i ++ )
      {
	Y[i] = Y[i-1] + dy[i-1];
	Y[Ncy-i] = Y[Npy-i] - dy[i-1];
      }    
  }
  else if ( ratioBCy == 1 ) {
    dyUni = Ly / Ncy;
    for ( int i = 1; i < Npy ; i ++ )
      {
	dy[i-1] = dyUni;
	Y[i] = Y[0] + i * dy[i-1]; 
      }
        
  }


  // z direction

  if ( ratioBCz != 1 ){
    dzcenter = MeshSize(Nczhalf,Lzhalf,ratioBCz);
    for ( int i = 0; i < Nczhalf; i++ ) {
      dz[i] = dzcenter * pow( ratioBCz, Nczhalf-1-i );
      dz[Ncz-i-1] = dz[i];
    }
    
    for ( int i = 1; i < Nczhalf; i ++ )
      {
	Z[i] = Z[i-1] + dz[i-1];
	Z[Ncz-i] = Z[Npz-i] - dz[i-1];
      }    
  }
  else if ( ratioBCz == 1 ){
    dzUni = Lz / Ncz;
    for ( int i = 1; i < Npz ; i ++ )
      {
	dz[i-1] = dzUni;
	Z[i] = Z[0] + i * dz[i-1]; 
      }    
    
  }


  double *ElementAreaTB = new double[TopBottomNo];
  double *ElementAreaFB = new double[FrontBackNo];
  double *ElementAreaLR = new double[LeftRightNo];
  double *ElementVol = new double[VolElementNo];

  double *ElementArea[6];
  ElementArea[0] = ElementAreaTB;
  ElementArea[1] = ElementAreaTB;
  ElementArea[2] = ElementAreaFB;
  ElementArea[3] = ElementAreaFB;
  ElementArea[4] = ElementAreaLR;
  ElementArea[5] = ElementAreaLR;
  
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


  X[Ncxhalf] = 0;
  Y[Ncyhalf] = 0;
  Z[Nczhalf] = 0;   
 

// initial as all volume elements ray no zeros first.
  // initial volume ray numbers
  // following x, y, z, so now from left to right, then front to back , and finally bottom to top
   for ( int k = 0; k < Ncz; k ++ )
     for ( int j = 0; j < Ncy; j ++ )
       for ( int i = 0; i < Ncx; i ++ )
	 rayNo_Vol[ i + j*Ncx + k*TopBottomNo] = 0; 

   for ( int k = int(Ncz/2)-1; k < int(Ncz/2)+1; k ++ )
      for ( int j = int(Ncy/2)-1; j < int(Ncy/2)+1; j ++ )
       for ( int i = 0; i < Ncx; i ++ )
	 rayNo_Vol[ i + j*Ncx + k*TopBottomNo] = 10000;   
   //  rayNo_Vol[454] = 100000;

   int iSurface;
   // initial all surface elements ray no = 0
   // top, bottom surfaces
   for ( int j = 0; j < Ncy; j ++ )
     for ( int i = 0; i < Ncx; i ++){
       iSurface = i + j*Ncx;
       rayNo_surface[TOP][iSurface] = 0;
       rayNo_surface[BOTTOM][iSurface] = 0;
     }

   // front back surfaces
   for ( int k = 0; k < Ncz; k ++ )
     for ( int i = 0; i < Ncx; i ++){
       iSurface = i + k*Ncx;
       rayNo_surface[FRONT][iSurface] = 0;
       rayNo_surface[BACK][iSurface] = 0;
     }   


   // left right surfaces
   for ( int k = 0; k < Ncz; k ++ )
     for ( int j = 0; j < Ncy; j ++){
       iSurface = j + k*Ncy;
       rayNo_surface[LEFT][iSurface] = 0;
       rayNo_surface[RIGHT][iSurface] = 0;
     }


     for ( int k = int(Ncz/2)-1; k < int(Ncz/2)+1; k ++ )
     for ( int j = 0; j < Ncy; j ++){
       iSurface = j + k*Ncy;
       rayNo_surface[LEFT][iSurface] = 10000;
       rayNo_surface[RIGHT][iSurface] = 10000;
      
     }

     
   MakeTableFunction obTable;    
   double *CO2 = new double [VolElementNo];
   double *H2O = new double [VolElementNo];
   double *SFV = new double [VolElementNo];
   
   // case set up-- dont put these upfront , put them here. otherwise return compile errors
   // when update anycases .cc, need to remove *.o, then recompile
   // otherwise,the input.cc file wont get updated in the binary files
   
   //  #include "inputBenchmark.cc"
   //  #include "inputBenchmarkSurf.cc"
   //   #include "inputNonblackSurf.cc"
   //  #include "inputScattering.cc"   
   //  #include "inputScatteringAniso.cc"
   //  #include "inputLiuWsgg.cc"
   //#include "inputBressloffRadCoeff.cc"
   cout <<" before inputFSK file" << endl;
   // #include "inputFSKhomoWebbRetawithIb.cc"
     #include "inputFSKhomoWebbRetanoIb.cc"
    
   
  int rayNouniform;
  rayNouniform = 10000;
  // generate uniform distributed from 0 to 1 , R same size as rayNo.
  int Runisize;
  Runisize = rayNouniform; // same as rayNo
  
  double dRuni, Rgg;
  
  dRuni = 1.0/Runisize; // has to be 1.0 otherwise return an integer.
  cout<< "Runisize = " << Runisize << endl;
  cout << "dRuni = " << dRuni << endl;
  
  // this section can be carried out in the loop for each element
  // when the rayNo are not the same for every element
  
  double *Runi = new double[Runisize];
  double *g = new double[Runisize];
  double *kl = new double[Runisize];
  double *Ibeta = new double[Runisize];
  double Tuni = 1000; // for homogeneous case
  double C1, C2;
  
  //%Ibeta = Ebeta/pi;
  //% Ebeta = C1 wvn^3 / ( exp(C2 * eta/T) - 1) -- W/m
  //% C1 = 2*pi*h*c0^2
  //% C2 = h*c0/k
  C1 = 3.7418e-16;// % [W m^2]
  C2 = 1.4388; //%[cm K]
  
  // make Runi not starting from 0, as R = 0-> g= 0, -> small k, not important rays
  for ( int i = 0; i < Runisize; i ++ ){
    Runi[i] = (i+1) * dRuni;
    // cout << "Runi = " << Runi[i] << endl;
    Rgg = Runi[i];
    // Since Runi is pre-set, we can pre-calculate g and kl from Runi.    
    obBST.search(Rgg, Rkg, gSize);
    obBST.calculate_gk(gk, Rkg, Rgg);
    g[i] = obBST.get_g(); // for Reta, convert wavenumber unit from cm-1 to m-1
    //  cout << " g= "<< g << endl;
    kl[i] = obBST.get_k()*100; // convert unit from cm-1 to m-1
    Ibeta[i] = C1 * g[i] *  g[i] *  g[i]  * 1e6 /
      ( exp( C2*  g[i] / Tuni )- 1) / pi;
    
  }

    //  obTable.singleArrayTable(kl_Vol, VolElementNo, 1, "klVolTablelast.dat");
  //  obTable.singleArrayTable(kl, Runisize, 1, "abcsTableReta10000IbCDF.dat");
  // obTable.singleArrayTable(g, Runisize, 1, "wvnTableReta10000IbCDF.dat");
  // obTable.singleArrayTable(Runi, Runisize, 1, "RuniTableReta10000IbCDF.dat");

   MTRand MTrng;
   VolElement obVol;
   VirtualSurface obVirtual;
   obVirtual.get_PhFunc(PhFunc, linear_b, eddington_f, eddington_g);   
   ray obRay(VolElementNo,Ncx, Ncy, Ncz);
   
   double OutIntenVol, traceProbability, LeftIntenFrac, sumIncomInten, aveIncomInten;
   double PathLeft, PathSurfaceLeft, weight;
   double previousSum, currentSum;
   double SurLeft;

   double theta, phi;
   double random1, random2;
   double s[3];
  
   double sumQsurface = 0;
   double sumQvolume = 0;
  
   // when ProcessorDone == 0, fresh start on a surface
   // when ProcessorDone == 1 , it is done with this processor calculation.
   // when ProcessorDone == 2, continue calculating onto another surface
   
   //   int local_Counter, ProcessorDone;  
   //   int my_rank_Start_No;
   
   // define these pointer arrays out of any if or loop,
   // otherwise they only exist within that domain instead of whole program
   
   RealSurface *RealPointer;
   
   TopRealSurface obTop_init;
   BottomRealSurface obBottom_init;
   FrontRealSurface obFront_init;
   BackRealSurface obBack_init;
   LeftRealSurface obLeft_init;
   RightRealSurface obRight_init;
   
   double *global_qsurface = new double [surfaceElementNo];
   double *global_Qsurface = new double [surfaceElementNo];  
   double *global_qdiv = new double [VolElementNo];
   double *global_Qdiv = new double [VolElementNo];


  // these values being used in both surface elements and control volumes
  int hitSurfaceIndex, hitSurfaceFlag;
  int surfaceFlag;
  int surfaceIndex;
  int rayCounter;

 //// =============================== Calculation starts ========================
  
  // when surfaces are cold, set local_rayNoSurface = 0

  int iIndex, jIndex, kIndex;
  int thisRayNo;
  RadWsgg obWsgg;
  int RkgIndex;
    

 
 
 // for Volume's  Intensity
   // for volume, use black intensity per wavenumber Ibeta,
  // note in I-eta, it is not the total black body intensity integrated over whole wavenumber
   for ( int i = 0; i < VolElementNo; i ++ )
     IntenArray_Vol[i] = 0; //obVol.VolumeIntensityBlack(i, T_Vol, a_Vol);

   // this is valid for gray surfaces, which in Webbhomo case is true.
   // for gray surfaces, emiss_surface doesnot change at different g ( or wavenumbers)
   // top bottom surfaces intensity
   for ( int i = 0;  i < TopBottomNo; i ++ ) {
     RealPointer = &obTop_init;
     IntenArray_surface[TOP][i] = RealPointer->SurfaceIntensity(i, emiss_surface[TOP],
							       T_surface[TOP],
							       a_surface[TOP]);
     
   }
   
   
  for ( int i = 0; i < TopBottomNo; i ++ ) {
    RealPointer = &obBottom_init;
    IntenArray_surface[BOTTOM][i] = RealPointer->SurfaceIntensity(i,
								 emiss_surface[BOTTOM],
								 T_surface[BOTTOM],
								 a_surface[BOTTOM]);
  }
  
  // front back surfaces intensity
  for ( int i = 0;  i < FrontBackNo; i ++ ) {
    RealPointer = &obFront_init;
    IntenArray_surface[FRONT][i] = RealPointer->SurfaceIntensity(i,
								emiss_surface[FRONT],
								T_surface[FRONT],
								a_surface[FRONT]);

  }

  
  for ( int i = 0; i < FrontBackNo; i ++ ) {
    RealPointer = &obBack_init;
    IntenArray_surface[BACK][i] = RealPointer->SurfaceIntensity(i,
							       emiss_surface[BACK],
							       T_surface[BACK],
							       a_surface[BACK]);
  }


  // left right surface intensity
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    RealPointer = &obLeft_init;
    IntenArray_surface[LEFT][i] = RealPointer->SurfaceIntensity(i,
							       emiss_surface[LEFT],
							       T_surface[LEFT],
							       a_surface[LEFT]);
  }

  
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    RealPointer = &obRight_init;
    IntenArray_surface[RIGHT][i] = RealPointer->SurfaceIntensity(i,
								emiss_surface[RIGHT],
								T_surface[RIGHT],
								a_surface[RIGHT]);
  }
 
  // end of recalculate Intensity
  
  time (&time_start);
  
  if ( rayNoSurface != 0 ) { // have rays emitting from surface elements


    // ------------- top surface -------------------
  surfaceFlag = TOP;
    
    // consider Z[] array from 0 to Npz -1 = Ncz
    kIndex = Ncz-1; // iIndex, jIndex, kIndex is consitent with control volume's index
    
    for ( jIndex = 0; jIndex < Ncy; jIndex ++ ) {
      for ( iIndex = 0; iIndex < Ncx; iIndex ++){
	//	cout << "iIndex = " << iIndex <<"; jIndex = " << jIndex << endl;
	surfaceIndex = iIndex + jIndex * Ncx;
	thisRayNo = rayNo_surface[surfaceFlag][surfaceIndex];

	if ( thisRayNo != 0 ) { // rays emitted from this surface
	  
	  MTrng.seed(surfaceIndex);
	  TopRealSurface obTop(iIndex, jIndex, kIndex, Ncx);
	  RealPointer = &obTop;
	  
	  // search for corresponding Rkg in file for the new ray
	  // then this ray will carry the energy from same wavenumber.
	  // the whole domain is set at this wavenumber

	  // Call BinarySearchTree, return kl_Vol
	  // count which g got called most.
	  
	  rayfromSurf(obTop,
		      RealPointer,
		      obVirtual,
		      obRay,
		      MTrng,
		      surfaceFlag,
		      surfaceIndex,
		      alpha_surface,
		      emiss_surface,
		      T_surface,
		      a_surface,
		      rs_surface,
		      rd_surface,
		      IntenArray_Vol,
		      IntenArray_surface,
		      X, Y, Z,
		      kl_Vol, scatter_Vol,
		      VolFeature,
		      thisRayNo,
		      iIndex,
		      jIndex,
		      kIndex,
		      iggNo,
		      StopLowerBound,
		      netInten_surface,
		      s,
		      obBST,
		      gSize, VolElementNo, Rkg, gk, kl, g,
		      Ibeta);
	 
	}

     } // end iIndex
   
  }// end jIndex

        // ----------------- end of top surface -----------------------
    // cout << "done with top" << endl;
    
    // ------------- bottom surface -------------------
  surfaceFlag = BOTTOM;
    
    // consider Z[] array from 0 to Npz -1 = Ncz
    kIndex = 0; // iIndex, jIndex, kIndex is consitent with control volume's index
    
    for ( jIndex = 0; jIndex < Ncy; jIndex ++ ) {
      for ( iIndex = 0; iIndex < Ncx; iIndex ++){
	//	cout << "iIndex = " << iIndex <<"; jIndex = " << jIndex << endl;
	surfaceIndex = iIndex + jIndex * Ncx;
	thisRayNo = rayNo_surface[surfaceFlag][surfaceIndex];

	if ( thisRayNo != 0 ) { // rays emitted from this surface
	  
	  MTrng.seed(surfaceIndex + BottomStartNo);
	  BottomRealSurface obBottom(iIndex, jIndex, kIndex, Ncx);
	  RealPointer = &obBottom;
	  
	  rayfromSurf(obBottom,
		      RealPointer,
		      obVirtual,
		      obRay,
		      MTrng,
		      surfaceFlag,
		      surfaceIndex,
		      alpha_surface,
		      emiss_surface,
		      T_surface,
		      a_surface,
		      rs_surface,
		      rd_surface,
		      IntenArray_Vol,
		      IntenArray_surface,
		      X, Y, Z,
		      kl_Vol, scatter_Vol,
		      VolFeature,
		      thisRayNo,
		      iIndex,
		      jIndex,
		      kIndex,
		      iggNo,
		      StopLowerBound,
		      netInten_surface,
		      s,
		      obBST,
		      gSize, VolElementNo, Rkg, gk, kl, g,
		      Ibeta);
	}


     } // end iIndex
   
  }// end jIndex

    // ----------------- end of bottom surface -----------------------
    //   cout << "done with bottom " << endl;



   // ------------- front surface -------------------
  surfaceFlag = FRONT;
    
    // consider Z[] array from 0 to Npz -1 = Ncz
    jIndex = 0; // iIndex, jIndex, kIndex is consitent with control volume's index
    
    for ( kIndex = 0; kIndex < Ncz; kIndex ++ ) {
      for ( iIndex = 0; iIndex < Ncx; iIndex ++){
	//	cout << "iIndex = " << iIndex <<"; jIndex = " << jIndex << endl;
	surfaceIndex = iIndex + kIndex * Ncx;
	thisRayNo = rayNo_surface[surfaceFlag][surfaceIndex];

	if ( thisRayNo != 0 ) { // rays emitted from this surface
	  
	  MTrng.seed(surfaceIndex + FrontStartNo);
	  FrontRealSurface obFront(iIndex, jIndex, kIndex, Ncx);
	  RealPointer = &obFront;
	  
	  rayfromSurf(obFront,
		      RealPointer,
		      obVirtual,
		      obRay,
		      MTrng,
		      surfaceFlag,
		      surfaceIndex,
		      alpha_surface,
		      emiss_surface,
		      T_surface,
		      a_surface,
		      rs_surface,
		      rd_surface,
		      IntenArray_Vol,
		      IntenArray_surface,
		      X, Y, Z,
		      kl_Vol, scatter_Vol,
		      VolFeature,
		      thisRayNo,
		      iIndex,
		      jIndex,
		      kIndex,
		      iggNo,
		      StopLowerBound,
		      netInten_surface,
		      s, obBST,
		      gSize, VolElementNo, Rkg, gk, kl, g,
		      Ibeta);
	}


     } // end iIndex
   
  }// end kIndex

    // ----------------- end of front surface -----------------------
    //   cout << "done with front " << endl;


  // ------------- back surface -------------------
  surfaceFlag = BACK;
    
    // consider Z[] array from 0 to Npz -1 = Ncz
    jIndex = Ncy-1; // iIndex, jIndex, kIndex is consitent with control volume's index
    
    for ( kIndex = 0; kIndex < Ncz; kIndex ++ ) {
      for ( iIndex = 0; iIndex < Ncx; iIndex ++){
	//	cout << "iIndex = " << iIndex <<"; jIndex = " << jIndex << endl;
	surfaceIndex = iIndex + kIndex * Ncx;
	thisRayNo = rayNo_surface[surfaceFlag][surfaceIndex];

	if ( thisRayNo != 0 ) { // rays emitted from this surface
	  
	  MTrng.seed(surfaceIndex + BackStartNo);
	  BackRealSurface obBack(iIndex, jIndex, kIndex, Ncx);
	  RealPointer = &obBack;
	  
	  rayfromSurf(obBack,
		      RealPointer,
		      obVirtual,
		      obRay,
		      MTrng,
		      surfaceFlag,
		      surfaceIndex,
		      alpha_surface,
		      emiss_surface,
		      T_surface,
		      a_surface,
		      rs_surface,
		      rd_surface,
		      IntenArray_Vol,
		      IntenArray_surface,
		      X, Y, Z,
		      kl_Vol, scatter_Vol,
		      VolFeature,
		      thisRayNo,
		      iIndex,
		      jIndex,
		      kIndex,
		      iggNo,
		      StopLowerBound,
		      netInten_surface,
		      s, obBST, 
		      gSize, VolElementNo, Rkg, gk, kl, g,
		      Ibeta);
	}


     } // end iIndex
   
  }// end kIndex

    // ----------------- end of back surface -----------------------
    // cout << "done with Back " << endl;

    

  // ------------- left surface -------------------
  surfaceFlag = LEFT;
    
    // consider Z[] array from 0 to Npz -1 = Ncz
    iIndex = 0; // iIndex, jIndex, kIndex is consitent with control volume's index
    
    for ( kIndex = 0; kIndex < Ncz; kIndex ++ ) {
      for ( jIndex = 0; jIndex < Ncy; jIndex ++){
	//	cout << "iIndex = " << iIndex <<"; jIndex = " << jIndex << endl;
	surfaceIndex = jIndex + kIndex * Ncy;
	thisRayNo = rayNo_surface[surfaceFlag][surfaceIndex];

	if ( thisRayNo != 0 ) { // rays emitted from this surface
	  
	  MTrng.seed(surfaceIndex + LeftStartNo);
	  LeftRealSurface obLeft(iIndex, jIndex, kIndex, Ncy);
	  RealPointer = &obLeft;

	  rayfromSurf(obLeft,
		      RealPointer,
		      obVirtual,
		      obRay,
		      MTrng,
		      surfaceFlag,
		      surfaceIndex,
		      alpha_surface,
		      emiss_surface,
		      T_surface,
		      a_surface,
		      rs_surface,
		      rd_surface,
		      IntenArray_Vol,
		      IntenArray_surface,
		      X, Y, Z,
		      kl_Vol, scatter_Vol,
		      VolFeature,
		      thisRayNo,
		      iIndex,
		      jIndex,
		      kIndex,
		      iggNo,
		      StopLowerBound,
		      netInten_surface,
		      s, obBST, 
		      gSize, VolElementNo, Rkg, gk, kl, g,
		      Ibeta);
	}


     } // end jIndex
   
  }// end kIndex

    // ----------------- end of left surface -----------------------
    // cout << "done with left " << endl;



   // ------------- right surface -------------------
  surfaceFlag = RIGHT;
    
    // consider Z[] array from 0 to Npz -1 = Ncz
    iIndex = Ncx-1; // iIndex, jIndex, kIndex is consitent with control volume's index
    
    for ( kIndex = 0; kIndex < Ncz; kIndex ++ ) {
      for ( jIndex = 0; jIndex < Ncy; jIndex ++){
	//	cout << "iIndex = " << iIndex <<"; jIndex = " << jIndex << endl;
	surfaceIndex = jIndex + kIndex * Ncy;
	thisRayNo = rayNo_surface[surfaceFlag][surfaceIndex];

	if ( thisRayNo != 0 ) { // rays emitted from this surface
	  
	  MTrng.seed(surfaceIndex + RightStartNo);
	  RightRealSurface obRight(iIndex, jIndex, kIndex, Ncy);
	  RealPointer = &obRight;

	  rayfromSurf(obRight,
		      RealPointer,
		      obVirtual,
		      obRay,
		      MTrng,
		      surfaceFlag,
		      surfaceIndex,
		      alpha_surface,
		      emiss_surface,
		      T_surface,
		      a_surface,
		      rs_surface,
		      rd_surface,
		      IntenArray_Vol,
		      IntenArray_surface,
		      X, Y, Z,
		      kl_Vol, scatter_Vol,
		      VolFeature,
		      thisRayNo,
		      iIndex,
		      jIndex,
		      kIndex,
		      iggNo,
		      StopLowerBound,
		      netInten_surface,
		      s, obBST,
		      gSize, VolElementNo, Rkg, gk, kl, g,
		      Ibeta);
	}


     } // end jIndex
   
  }// end kIndex

    // ----------------- end of right surface -----------------------
    //  cout << "done with right " << endl;
  
} // end if rayNoSurface ! = 0 ?
   
	  
  iggNo = 0;
  
  //  cout << " i am here after one iggNo" << endl;
  if ( rayNoVol != 0 ) { // emitting ray from volume
    //    cout << "start from Vol" << endl;
    int rayCounter, VolIndex;
    
    for ( int kVolIndex = 0; kVolIndex < Ncz; kVolIndex ++ ) {
      for ( int jVolIndex = 0 ; jVolIndex < Ncy; jVolIndex ++ ) {
	for ( int iVolIndex = 0; iVolIndex < Ncx; iVolIndex ++ ) {

	  VolIndex = iVolIndex + jVolIndex * Ncx + kVolIndex * TopBottomNo;

	  if ( rayNo_Vol[VolIndex] != 0 ) {
	    
	    MTrng.seed(VolIndex);	    
	    VolElement obVol(iVolIndex, jVolIndex, kVolIndex, Ncx, Ncy);
	    

	    double *IncomingIntenVol = new double [ rayNo_Vol[VolIndex] ];
	  
	    for ( rayCounter = 0; rayCounter < rayNo_Vol[VolIndex]; rayCounter ++) {

	      /*
	      Rgg = Runi[rayCounter];
	      // Rgg =  MTrng.randExc(); // Rgg random number
	      //	  cout << " Rgg = " << Rgg << endl;
	      obBST.search(Rgg, Rkg, gSize);
	      obBST.calculate_gk(gk, Rkg, Rgg);
	      g = obBST.get_g();
	      //  cout << " g= "<< g << endl;
	      kl = obBST.get_k();
	      //  cout << "kl = " << kl << endl;
	      countg[obBST.get_lowI()] ++;
	      */

	      
	      // set absorption coeff
	      for ( int ki = 0; ki < VolElementNo; ki++){
		IntenArray_Vol[ki] = Ibeta[rayCounter];
		kl_Vol[ki] = kl[rayCounter];
	      }
	      
	      OutIntenVol = IntenArray_Vol[VolIndex] * kl_Vol[VolIndex];
	      
	      LeftIntenFrac = 1;
	      weight = 1;
	      traceProbability = 1;
	      previousSum = 0;
	      currentSum = 0;
	      IncomingIntenVol[rayCounter] = 0;
	      
	      // when absorbed by this emitting volume, only absorbed by kl_Vol portion.
	      SurLeft = kl_Vol[VolIndex];
	      
	      // get emitting ray's direction vector s
	      obRay.set_emissS_vol(MTrng, s);
	      
	      obRay.set_directionS(s); // put s into directionVector ( private )
	      obVol.get_limits(X, Y, Z);
	      
	      // VolIndex is the vIndex is
	      // VoliIndex + VoljIndex * Ncx + VolkIndex * TopBottomNo
	      
	      obRay.set_emissP(MTrng,
			       obVol.get_xlow(), obVol.get_xup(),
			       obVol.get_ylow(), obVol.get_yup(),
			       obVol.get_zlow(), obVol.get_zup());

	      //   obRay.set_straightP();
	      //   obRay.set_straight_len(0);
	      
	      obRay.set_currentvIndex(iVolIndex, jVolIndex, kVolIndex);
	      
	      // emitting rays from volume, then surely has participating media
	      
	      // only one criteria for now ( the left energy percentage )
	      //   vectorIndex = 0;

	      obRay.dirChange = 1;
	      
	      do {
		weight = weight / traceProbability;
		
		previousSum = currentSum;
		
		obRay.TravelInMediumInten(MTrng, obVirtual,
					  kl_Vol, scatter_Vol,
					  X, Y, Z, VolFeature,
					  PathLeft, PathSurfaceLeft);
		
		
		// the upper bound of the segment
		currentSum = previousSum + PathLeft;
		
		// the IntensityArray for volumes are black ones.
		// use the previous SurLeft here.
		// SurLeft is not updated yet.
		
		IncomingIntenVol[rayCounter] = IncomingIntenVol[rayCounter] + 
		  IntenArray_Vol[obRay.get_currentvIndex()]
		  * ( exp(-previousSum) - exp(-currentSum) ) * SurLeft
		  * weight;
		
		// SurLeft to accout for the real surface absorption effect on intensity
		
		if ( !obRay.VIRTUAL ) {
		  
		  hitSurfaceFlag = obRay.get_surfaceFlag();
		  hitSurfaceIndex = obRay.get_hitSurfaceIndex();
		  
		  // PathSurfaceLeft is updated here
		  // and it comes into effect for next travelling step.
		  obRay.hitRealSurfaceInten(MTrng,
					    alpha_surface[hitSurfaceFlag],
					    rs_surface[hitSurfaceFlag],
					    rd_surface[hitSurfaceFlag],
					    PathSurfaceLeft);
		  
		  IncomingIntenVol[rayCounter] = IncomingIntenVol[rayCounter] +
		    IntenArray_surface[hitSurfaceFlag][hitSurfaceIndex] *
		    exp ( -currentSum ) * SurLeft
		    * weight;
		  
		}
		
		// set hitPoint as new emission Point
		// and direction of the ray already updated
		obRay.update_emissP();
		obRay.update_vIndex();
		
		SurLeft = SurLeft * PathSurfaceLeft;		
		LeftIntenFrac = exp(-currentSum) * SurLeft;
		traceProbability = min(1.0, LeftIntenFrac/StopLowerBound);

		
	      }while ( MTrng.randExc() < traceProbability ); // continue the path
	      
	      // temperory solution.
	      // otherwise get an array of OutIntenVol[rayCounter]
	      IncomingIntenVol[rayCounter] = OutIntenVol - IncomingIntenVol[rayCounter];
	      
	    } // rayCounter loop
	    
	  
	    // deal with the current control volume
	    // isotropic emission, weighting factors are all the same on all directions
	    // net = OutInten - averaged_IncomingIntenDir
	    // div q = 4 * pi * netInten

	    // integrate over all g of incoming Intensity
	    // here the rayNo_Vol[VolIndex] has to be the same as Runisize

	    // the OutIntenVol is changing with each ray too!!!
	    sumIncomInten = 0;

	    //    if ( VolIndex==454) 
	    //      obTable.twoArrayTable( rayNouniform, g, IncomingIntenVol, "Ieta10000cell454noIbCDF-L5.dat");

	 	    
	    for ( int aaa = 0; aaa < rayNo_Vol[VolIndex]-1 ; aaa ++ )
	      sumIncomInten = sumIncomInten +
		( IncomingIntenVol[aaa+1] + IncomingIntenVol[aaa] ) *
		( g[aaa+1] - g[aaa]) * 100 /2.0;
	    
	    
	    delete[] IncomingIntenVol;
	    
	    // aveIncomInten = sumIncomInten / rayNo_Vol[VolIndex];
	    // cout << "aveIncomInten = " << aveIncomInten << endl;
	    
	    // netInten_Vol[VolIndex][iggNo] = OutIntenVol - aveIncomInten;

	     netInten_Vol[VolIndex][iggNo] = sumIncomInten;
	    
	  } // if rayNo_Vol[VolIndex] != 0
	  
	  
	} // end if iVolIndex
	
      } // end if jVolIndex
      
    } // end if kVolIndex

  
  } // end rayNoVol!= 0 
    

  

  time (&time_end); 
  
  // surface cell
    
  double *integrIntenSurface[6];
  for ( int i = 0; i < 6; i ++)
    integrIntenSurface[i] = new double[surfaceNo[i]];

  if ( rayNoSurface != 0 ) {
    
  for ( int i = 0; i < 6; i ++)
    for ( int j = 0; j < surfaceNo[i]; j++)
      integrIntenSurface[i][j] = 0;
  
  // WSGG all weights are 1, sum all bands together

  for ( surfaceFlag = 0; surfaceFlag < 6; surfaceFlag++)
    for ( int elementNo = 0; elementNo < surfaceNo[surfaceFlag]; elementNo ++)
      for ( int iggNo = 0; iggNo < 1; iggNo ++)
	integrIntenSurface[surfaceFlag][elementNo] = integrIntenSurface[surfaceFlag][elementNo] +
	  netInten_surface[surfaceFlag][elementNo][iggNo];

  
  iSurface = 0;
  for ( surfaceFlag = 0; surfaceFlag < 6; surfaceFlag ++ ) {
    for ( surfaceIndex = 0; surfaceIndex < surfaceNo[surfaceFlag]; surfaceIndex ++) {
      global_qsurface[iSurface] = pi * integrIntenSurface[surfaceFlag][surfaceIndex];
      global_Qsurface[iSurface] = global_qsurface[iSurface] * ElementArea[surfaceFlag][surfaceIndex];
       sumQsurface = sumQsurface + global_Qsurface[iSurface] ;
      iSurface ++;
    }
  }
  
  /*
    obTable.vtkSurfaceTableMake("vtkSurfaceWebbHomoReta10000-L1-202020IbCDF-11111", Npx, Npy, Npz,
			      X, Y, Z, surfaceElementNo,
			      global_qsurface, global_Qsurface);
  */

    }
    
  
    // Vol cell
  
   double integrIntenVol[VolElementNo];
   for ( int i = 0; i < VolElementNo; i ++)
     integrIntenVol[i] = 0;

   if ( rayNoVol != 0 ){
     
  // WSGG all weights are 1, sum all bands together

    for ( int i = 0; i < VolElementNo; i ++)
      for ( int iggNo = 0; iggNo < 1; iggNo ++)
	integrIntenVol[i]= integrIntenVol[i]+
	  netInten_Vol[i][iggNo] ;

  
  
  for (int i = 0 ; i < VolElementNo; i ++ ) {
    global_qdiv[i] = 4 * pi * integrIntenVol[i];
    global_Qdiv[i] = global_qdiv[i] * ElementVol[i];
    sumQvolume = sumQvolume + global_Qdiv[i];
  }
  
  obTable.vtkVolTableMake("vtkVolWebbHomoReta10000-L1-202020NoIbCDF-11111",
			  Npx, Npy, Npz,
			  X, Y, Z, VolElementNo,
			  global_qdiv, global_Qdiv);

  
   }
   
  cout << "sumQsurface = " << sumQsurface << endl;
  cout << "sumQvolume = " << sumQvolume << endl;
  
  double difference, Frac, timeused;
  difference = sumQsurface + sumQvolume;
  cout << " the heat balance difference = (sumQsurface + sumQvolume) = " <<difference << endl;
  
  Frac = difference / sumQsurface;
  
  cout << " Frac = " << Frac << endl;
  
  cout << " Lx = " << Lx << "; Ly = " << Ly << " ; Lz = " << Lz << endl;
  cout << " Ncx = " << Ncx << " ; Ncy = " << Ncy << "; Ncz = " << Ncz << endl;
  cout << " ratioBCx = " << ratioBCx << "; ratioBCy = " << ratioBCy <<
    "; ratioBCz = " << ratioBCz << endl;
  
  timeused = difftime (time_end,time_start);
  
  cout << " time used up (S) = " << timeused << "sec." << endl;
  cout << "iggNo = " << iggNo << endl;
  
  delete[] T_Vol;
  delete[] kl_Vol;   
  delete[] scatter_Vol;
  delete[] rayNo_Vol;
  delete[] IntenArray_Vol;
  delete[] a_Vol;
  delete[] netInten_Vol;
  /*
  delete[] a_surface;
  delete[] T_surface;
  delete[] rs_surface;
  delete[] emiss_surface;
  delete[] rd_surface;
  delete[] netInten_surface;
  delete[] alpha_surface;
  delete[] rayNo_surface;
  */
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
  delete[] VolFeature;
  
  delete[] global_qdiv;
  delete[] global_Qdiv;
  delete[] global_qsurface;
  delete[] global_Qsurface;

  delete[] CO2;
  delete[] H2O;
  delete[] SFV;
  delete[] Rkg;
  delete[] gk;
  delete[] g;
  delete[] kl;
  delete[] Runi;
  delete[] Ibeta;
  //  delete[]integrIntenSurface; 
  //   MPI_Finalize();

  return 0;


}

