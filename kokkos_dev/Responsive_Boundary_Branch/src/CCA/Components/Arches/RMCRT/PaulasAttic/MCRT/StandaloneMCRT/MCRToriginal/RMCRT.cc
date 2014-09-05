/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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

extern void
setupBenchmark( int Ncx, int Ncy, int Ncz, // number of cells in x, y, z directions
                double * T_Vol, 
                double * X, double * Y, double * Z,
                double * kl_Vol, double * a_Vol, double * scatter_Vol,
                int TopBottomNo, int FrontBackNo, int LeftRightNo,
                double * a_top_surface,     double * a_bottom_surface, double * a_front_surface, double * a_back_surface, double * a_left_surface, double * a_right_surface,
                double * rs_top_surface,    double * rs_bottom_surface, double * rs_front_surface, double * rs_back_surface, double * rs_left_surface, double * rs_right_surface,
                double * rd_top_surface,    double * rd_bottom_surface, double * rd_front_surface, double * rd_back_surface, double * rd_left_surface, double * rd_right_surface,
                double * alpha_top_surface, double * alpha_bottom_surface, double * alpha_front_surface, double * alpha_back_surface, double * alpha_left_surface, double * alpha_right_surface,
                double * emiss_top_surface, double * emiss_bottom_surface, double * emiss_front_surface, double * emiss_back_surface, double * emiss_left_surface, double * emiss_right_surface,
                double * T_top_surface,     double * T_bottom_surface, double * T_front_surface, double * T_back_surface, double * T_left_surface, double * T_right_surface);


int
main(int argc, char *argv[]){

  
//   int my_rank; // rank of process
//   int np; // number of processes
  time_t time_start, time_end;
  time (&time_start);
  

//   // starting up MPI
//   MPI_Init(&argc, &argv);
//   MPI_Barrier(MPI_COMM_WORLD);
  
//   precision = MPI_Wtick();
  
//   time1 = MPI_Wtime();
  
//   // Find out process rank
//    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

//   // Find out number of processes
//   MPI_Comm_size(MPI_COMM_WORLD, &np);


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
  double EnergyAmount; // set as customer self-set-up later
  double sumIncomInten, aveIncomInten;  


  rayNoSurface = 1;
  rayNoVol = 1;  
  Ncx = 10;
  Ncy = 10;
  Ncz = 10;
  ratioBCx = 1;
  ratioBCy = 1;
  ratioBCz = 1;
  Lx = 1;
  Ly = 1;
  Lz = 1;

  
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
  
  int ghostTotalNo =  ghostX * ghostY * ghostZ;
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

  // in Arches, do we have only one set of surface variables, or
  // six for six surfaces?????
  
  // get property of real surface
  // without table, need to initialize for each side of surfaces
  // then will loop over X, Y, Z to set property
  // so even the geometry is irregular, with X, Y, Z arrays, this should work too
  double *T_top_surface = new double [TopBottomNo];
  double *T_bottom_surface = new double [TopBottomNo];
  double *T_front_surface = new double [FrontBackNo];
  double *T_back_surface = new double [FrontBackNo];
  double *T_left_surface = new double [LeftRightNo];
  double *T_right_surface = new double [LeftRightNo];

  // alpha  = emiss
  // alpha + rs + rd = 1 for non transmittive surface
  
  // absorption coefficient for surfaces
  double *alpha_top_surface = new double [TopBottomNo];
  double *alpha_bottom_surface = new double [TopBottomNo];
  double *alpha_front_surface = new double [FrontBackNo];
  double *alpha_back_surface = new double [FrontBackNo];
  double *alpha_left_surface = new double [LeftRightNo];
  double *alpha_right_surface = new double [LeftRightNo];
  
  // specular reflective coefficient
  double *rs_top_surface = new double [TopBottomNo];
  double *rs_bottom_surface = new double [TopBottomNo];
  double *rs_front_surface = new double [FrontBackNo];
  double *rs_back_surface = new double [FrontBackNo];
  double *rs_left_surface = new double [LeftRightNo];
  double *rs_right_surface = new double [LeftRightNo];

  // diffusive reflective coefficient
  double *rd_top_surface = new double [TopBottomNo];
  double *rd_bottom_surface = new double [TopBottomNo];
  double *rd_front_surface = new double [FrontBackNo];
  double *rd_back_surface = new double [FrontBackNo];
  double *rd_left_surface = new double [LeftRightNo];
  double *rd_right_surface = new double [LeftRightNo];

  // emissive coefficient
  double *emiss_top_surface = new double [TopBottomNo];
  double *emiss_bottom_surface = new double [TopBottomNo];
  double *emiss_front_surface = new double [FrontBackNo];
  double *emiss_back_surface = new double [FrontBackNo];
  double *emiss_left_surface = new double [LeftRightNo];
  double *emiss_right_surface = new double [LeftRightNo];

  // the streching factor a for FSCK and FSSK boundary conditions
  double *a_top_surface = new double [TopBottomNo];
  double *a_bottom_surface = new double [TopBottomNo];
  double *a_front_surface = new double [FrontBackNo];
  double *a_back_surface = new double [FrontBackNo];
  double *a_left_surface = new double [LeftRightNo];
  double *a_right_surface = new double [LeftRightNo];
  
  // ray number for surfaces
  int *rayNo_top_surface = new int [TopBottomNo];
  int *rayNo_bottom_surface = new int [TopBottomNo];
  int *rayNo_front_surface = new int [FrontBackNo];
  int *rayNo_back_surface = new int [FrontBackNo];
  int *rayNo_left_surface = new int [LeftRightNo];
  int *rayNo_right_surface = new int [LeftRightNo];

  // IntenArray_surface ( outgoing intensity ) 
  double *IntenArray_top_surface = new double [TopBottomNo];
  double *IntenArray_bottom_surface = new double [TopBottomNo];
  double *IntenArray_front_surface = new double [FrontBackNo];
  double *IntenArray_back_surface = new double [FrontBackNo];
  double *IntenArray_left_surface = new double [LeftRightNo];
  double *IntenArray_right_surface = new double [LeftRightNo];

  // netInten_surface
  double *netInten_top_surface = new double [TopBottomNo];
  double *netInten_bottom_surface = new double [TopBottomNo];
  double *netInten_front_surface = new double [FrontBackNo];
  double *netInten_back_surface = new double [FrontBackNo];
  double *netInten_left_surface = new double [LeftRightNo];
  double *netInten_right_surface = new double [LeftRightNo];

  
  // get property of vol
  double *T_Vol = new double [VolElementNo];
  double *kl_Vol = new double [VolElementNo];
  double *scatter_Vol = new double [VolElementNo];
  double *a_Vol = new double[VolElementNo];
  int *rayNo_Vol = new int [VolElementNo];  
  double *IntenArray_Vol = new double [VolElementNo];
  double *netInten_Vol = new double [VolElementNo];
  

  // six surfaces
  double *alpha_surface[6], *rs_surface[6], *rd_surface[6], *IntenArray_surface[6];
  double *a_surface[6], *T_surface[6], *emiss_surface[6],  *netInten_surface[6];
  int *rayNo_surface[6];
  
  alpha_surface[0] = alpha_top_surface;
  alpha_surface[1] = alpha_bottom_surface;
  alpha_surface[2] = alpha_front_surface;
  alpha_surface[3] = alpha_back_surface;
  alpha_surface[4] = alpha_left_surface;
  alpha_surface[5] = alpha_right_surface;
  
  rs_surface[0] = rs_top_surface;
  rs_surface[1] = rs_bottom_surface;
  rs_surface[2] = rs_front_surface;
  rs_surface[3] = rs_back_surface;
  rs_surface[4] = rs_left_surface;
  rs_surface[5] = rs_right_surface;
    
  rd_surface[0] = rd_top_surface;
  rd_surface[1] = rd_bottom_surface;
  rd_surface[2] = rd_front_surface;
  rd_surface[3] = rd_back_surface;
  rd_surface[4] = rd_left_surface;
  rd_surface[5] = rd_right_surface;

  T_surface[0] = T_top_surface;
  T_surface[1] = T_bottom_surface;
  T_surface[2] = T_front_surface;
  T_surface[3] = T_back_surface;
  T_surface[4] = T_left_surface;
  T_surface[5] = T_right_surface;
    
  emiss_surface[0] = emiss_top_surface;
  emiss_surface[1] = emiss_bottom_surface;
  emiss_surface[2] = emiss_front_surface;
  emiss_surface[3] = emiss_back_surface;
  emiss_surface[4] = emiss_left_surface;
  emiss_surface[5] = emiss_right_surface;

  a_surface[0] = a_top_surface;
  a_surface[1] = a_bottom_surface;
  a_surface[2] = a_front_surface;
  a_surface[3] = a_back_surface;
  a_surface[4] = a_left_surface;
  a_surface[5] = a_right_surface;
  
  rayNo_surface[0] = rayNo_top_surface;
  rayNo_surface[1] = rayNo_bottom_surface;
  rayNo_surface[2] = rayNo_front_surface;
  rayNo_surface[3] = rayNo_back_surface;
  rayNo_surface[4] = rayNo_left_surface;
  rayNo_surface[5] = rayNo_right_surface;
   
  IntenArray_surface[0] = IntenArray_top_surface;
  IntenArray_surface[1] = IntenArray_bottom_surface;
  IntenArray_surface[2] = IntenArray_front_surface;
  IntenArray_surface[3] = IntenArray_back_surface;
  IntenArray_surface[4] = IntenArray_left_surface;
  IntenArray_surface[5] = IntenArray_right_surface;
    
  netInten_surface[0] = netInten_top_surface;
  netInten_surface[1] = netInten_bottom_surface;
  netInten_surface[2] = netInten_front_surface;
  netInten_surface[3] = netInten_back_surface;
  netInten_surface[4] = netInten_left_surface;
  netInten_surface[5] = netInten_right_surface;
    
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
	 rayNo_Vol[ i + j*Ncx + k*TopBottomNo] = 500; 
   // TopBottomNo = Ncx * Ncy;


   int iSurface;
   // initial all surface elements ray no = 0
   // top, bottom surfaces
   for ( int j = 0; j < Ncy; j ++ )
     for ( int i = 0; i < Ncx; i ++){
       iSurface = i + j*Ncx;
       rayNo_top_surface[iSurface] = 500;
       rayNo_bottom_surface[iSurface] = 500;
     }

   // front back surfaces
   for ( int k = 0; k < Ncz; k ++ )
     for ( int i = 0; i < Ncx; i ++){
       iSurface = i + k*Ncx;
       rayNo_front_surface[iSurface] = 500;
       rayNo_back_surface[iSurface] = 500;
     }   


   // left right surfaces
   for ( int k = 0; k < Ncz; k ++ )
     for ( int j = 0; j < Ncy; j ++){
       iSurface = j + k*Ncy;
       rayNo_left_surface[iSurface] = 500;
       rayNo_right_surface[iSurface] = 500;
     }

   // case set up-- dont put these upfront , put them here. otherwise return compile errors
   setupBenchmark( Ncx, Ncy, Ncz, // number of cells in x, y, z directions
                   T_Vol, 
                   X,  Y,  Z,
                   kl_Vol,  a_Vol,  scatter_Vol,
                   TopBottomNo, FrontBackNo, LeftRightNo,
                   a_top_surface,      a_bottom_surface,  a_front_surface,  a_back_surface,  a_left_surface,  a_right_surface,
                   rs_top_surface,     rs_bottom_surface,  rs_front_surface,  rs_back_surface,  rs_left_surface,  rs_right_surface,
                   rd_top_surface,     rd_bottom_surface,  rd_front_surface,  rd_back_surface,  rd_left_surface,  rd_right_surface,
                   alpha_top_surface,  alpha_bottom_surface,  alpha_front_surface,  alpha_back_surface,  alpha_left_surface,  alpha_right_surface,
                   emiss_top_surface,  emiss_bottom_surface,  emiss_front_surface,  emiss_back_surface,  emiss_left_surface,  emiss_right_surface,
                   T_top_surface,      T_bottom_surface,  T_front_surface,  T_back_surface,  T_left_surface,  T_right_surface);

   // #include "inputBenchmarkSurf.cc"
   // #include "inputNonblackSurf.cc"
   // #include "inputScattering.cc"   

   MTRand MTrng;
   VolElement obVol;
   
   double OutIntenVol, OutIntenSur, IntenFrac, LeftIntenFrac, pathlen;
   double PathLeft, PathSurfaceLeft;
   double previousSum, currentSum;
   double SurLeft;


   IntenFrac = 1e-10; //1e-20; // the percentage of Intensity left
   
   //   srand48 ( time ( NULL )); // Set the random number seed for drand48

   ray obRay(VolElementNo,Ncx, Ncy, Ncz);
   
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
   
   // for Volume's  Intensity
   // for volume, use black intensity
   for ( int i = 0; i < VolElementNo; i ++ )
     IntenArray_Vol[i] = obVol.VolumeIntensityBlack(i, T_Vol, a_Vol);
   
   // top bottom surfaces intensity
   for ( int i = 0;  i < TopBottomNo; i ++ ) {
     RealPointer = &obTop_init;
     IntenArray_top_surface[i] = RealPointer->SurfaceIntensity(i, emiss_top_surface,
							       T_top_surface,
							       a_top_surface);
     
   }
   
   
  for ( int i = 0; i < TopBottomNo; i ++ ) {
    RealPointer = &obBottom_init;
    IntenArray_bottom_surface[i] = RealPointer->SurfaceIntensity(i,
								 emiss_bottom_surface,
								 T_bottom_surface,
								 a_bottom_surface);
  }
  
  // front back surfaces intensity
  for ( int i = 0;  i < FrontBackNo; i ++ ) {
    RealPointer = &obFront_init;
    IntenArray_front_surface[i] = RealPointer->SurfaceIntensity(i,
								emiss_front_surface,
								T_front_surface,
								a_front_surface);

  }

  
  for ( int i = 0; i < FrontBackNo; i ++ ) {
    RealPointer = &obBack_init;
    IntenArray_back_surface[i] = RealPointer->SurfaceIntensity(i,
							       emiss_back_surface,
							       T_back_surface,
							       a_back_surface);
  }


  // left right surface intensity
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    RealPointer = &obLeft_init;
    IntenArray_left_surface[i] = RealPointer->SurfaceIntensity(i,
							       emiss_left_surface,
							       T_left_surface,
							       a_left_surface);
  }

  
  for ( int i = 0; i < LeftRightNo; i ++ ) {
    RealPointer = &obRight_init;
    IntenArray_right_surface[i] = RealPointer->SurfaceIntensity(i,
								emiss_right_surface,
								T_right_surface,
								a_right_surface);
  }
 

  // these values being used in both surface elements and control volumes
  int hitSurfaceIndex, hitSurfaceFlag;
  double alpha;
  int surfaceFlag;
  int surfaceIndex;
  int rayCounter;
  MakeTableFunction obTable;    
 //// =============================== Calculation starts ========================
  
  // when surfaces are cold, set local_rayNoSurface = 0

  int iIndex, jIndex, kIndex;
  int thisRayNo;
  
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

	    // get surface element's absorption coefficient
	  alpha = alpha_surface[surfaceFlag][surfaceIndex];	  
	  OutIntenSur = RealPointer->SurfaceIntensity(surfaceIndex,
						      emiss_surface[surfaceFlag],
						      T_surface[surfaceFlag],
						      a_surface[surfaceFlag]);
	  //	  cout << "OutIntenSur = " << OutIntenSur << endl;
	  
	  double *IncomingIntenSur = new double[ thisRayNo ];
	
	  // loop over ray numbers on each surface element
	  for ( rayCounter = 0; rayCounter < thisRayNo; rayCounter++ ) {
	    
	    LeftIntenFrac = 1;
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
			     obTop.get_xlow(), obTop.get_xup(),
			     obTop.get_ylow(), obTop.get_yup(),
			     obTop.get_zlow(), obTop.get_zup());
	    
	    obRay.set_directionS(s);
	    obRay.set_currentvIndex(iIndex, jIndex, kIndex);
	    
	    do {

	      previousSum = currentSum;

	      // checking scattering first
	      // if hit on virtual surface, PathSurfaceLeft is updated.
	      // else no update on PathSurfaceLeft.
	      obRay.TravelInMediumInten(MTrng,
					kl_Vol, scatter_Vol,
					X, Y, Z, VolFeature,
					PathLeft, PathSurfaceLeft);
	      

	      // the upper bound of the segment
	      currentSum = previousSum + PathLeft;
	      
	      IncomingIntenSur[rayCounter] = IncomingIntenSur[rayCounter] + 
		IntenArray_Vol[obRay.get_currentvIndex()] 
		* ( exp(-previousSum) - exp(-currentSum) ) * SurLeft;
	      
	   //    cout << "previousSum = " << previousSum << endl;
// 	      cout << "currentSum = " << currentSum << endl;
	      
// 	      	cout << "InComing = " << IncomingIntenSur[rayCounter] << endl;
// 	      cout << "IntenArray_Vol = " << IntenArray_Vol[obRay.get_currentvIndex()] << endl;
	      	      
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
		
// 		cout << "obRay.get_surfaceFlag() = " << obRay.get_surfaceFlag() << endl;
// 		cout << "surfaceFlag = " << surfaceFlag << endl;
		
		IncomingIntenSur[rayCounter] = IncomingIntenSur[rayCounter] +
		  IntenArray_surface[hitSurfaceFlag][hitSurfaceIndex] *
		  exp ( -currentSum ) * SurLeft;
		//	cout << "InComing = " << IncomingIntenSur[rayCounter] << endl;
	      }
	     

	     // set hitPoint as new emission Point
	     // and direction of the ray already updated
	      obRay.update_emissP();
	      obRay.update_vIndex();
	      
	      SurLeft = SurLeft * PathSurfaceLeft;
	      
	      LeftIntenFrac = exp( -currentSum) * SurLeft;
	      

	   }while ( LeftIntenFrac >= IntenFrac);
	 
	  } // rayCounter loop
	  

	  sumIncomInten = 0;
	  for ( int aaa = 0; aaa < thisRayNo; aaa ++ )
	    sumIncomInten = sumIncomInten + IncomingIntenSur[aaa];
	  //	  cout << "sumIncomInten = " << sumIncomInten << endl;
	  delete[] IncomingIntenSur;
	  
	  aveIncomInten = sumIncomInten / rayNo_surface[surfaceFlag][surfaceIndex];
	  
	  netInten_surface[surfaceFlag][surfaceIndex] =
	    OutIntenSur - aveIncomInten;

	  //	  cout << "netInten_surface = " << netInten_surface[surfaceFlag][surfaceIndex] << endl;
	 	  
	} // if thisRayNo != 0 
       
 
     } // end iIndex
   
  }// end jIndex
  
} // end if rayNoSurface ! = 0 ?
   
	  

  // surface cell
 
  iSurface = 0;
  for ( surfaceFlag = 0; surfaceFlag < 6; surfaceFlag ++ ) {
    for ( surfaceIndex = 0; surfaceIndex < surfaceNo[surfaceFlag]; surfaceIndex ++) {
      global_qsurface[iSurface] = pi * netInten_surface[surfaceFlag][surfaceIndex];
      global_Qsurface[iSurface] = global_qsurface[iSurface] * ElementArea[surfaceFlag][surfaceIndex];
       sumQsurface = sumQsurface + global_Qsurface[iSurface] ;
      iSurface ++;
    }
  }
  

  
  obTable.vtkSurfaceTableMake("vtkSurfaceNoTable", Npx, Npy, Npz,
			      X, Y, Z, surfaceElementNo,
			      global_qsurface, global_Qsurface);       

  
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

	    // VolIndex = obVol.get_VolIndex();
	    
	    OutIntenVol = obVol.VolumeIntensity(VolIndex,
						kl_Vol, T_Vol, a_Vol);
	    
	    double *IncomingIntenVol = new double [ rayNo_Vol[VolIndex] ];
	  
	    for ( rayCounter = 0; rayCounter < rayNo_Vol[VolIndex]; rayCounter ++) {
	      
	      LeftIntenFrac = 1;
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
	      
	      obRay.set_currentvIndex(iVolIndex, jVolIndex, kVolIndex);
	      
	      // emitting rays from volume, then surely has participating media
	      
	      // only one criteria for now ( the left energy percentage )
	      //   vectorIndex = 0;
	      
	      do {
		
		previousSum = currentSum;
		
		obRay.TravelInMediumInten(MTrng,
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
		  * ( exp(-previousSum) - exp(-currentSum) ) * SurLeft;
		
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
		    exp ( -currentSum ) * SurLeft;
		  
		}
		
		// set hitPoint as new emission Point
		// and direction of the ray already updated
		obRay.update_emissP();
		obRay.update_vIndex();
		
		SurLeft = SurLeft * PathSurfaceLeft;		
		LeftIntenFrac = exp(-currentSum) * SurLeft;
		
	      }while ( LeftIntenFrac >= IntenFrac);
	      


	    } // rayCounter loop
	    
	  
	    // deal with the current control volume
	    // isotropic emission, weighting factors are all the same on all directions
	    // net = OutInten - averaged_IncomingIntenDir
	    // div q = 4 * pi * netInten
	    
	    sumIncomInten = 0;
	    for ( int aaa = 0; aaa < rayNo_Vol[VolIndex]; aaa ++ )
	      sumIncomInten = sumIncomInten + IncomingIntenVol[aaa];
	    
	    
	    delete[] IncomingIntenVol;
	    
	    aveIncomInten = sumIncomInten / rayNo_Vol[VolIndex];
	    // cout << "aveIncomInten = " << aveIncomInten << endl;
	    
	    netInten_Vol[VolIndex] = OutIntenVol - aveIncomInten;
	    
	    
	  } // if rayNo_Vol[VolIndex] != 0
	  
	  
	} // end if iVolIndex
	
      } // end if jVolIndex
      
    } // end if kVolIndex

  } // end rayNoVol!= 0 
    



  
  
  // Vol cell

  for (int i = 0 ; i < VolElementNo; i ++ ) {
    global_qdiv[i] = 4 * pi * netInten_Vol[i];
    global_Qdiv[i] = global_qdiv[i] * ElementVol[i];
    sumQvolume = sumQvolume + global_Qdiv[i];
  }
  
  obTable.vtkVolTableMake("vtkVolBenchmarkNoTable",
			  Npx, Npy, Npz,
			  X, Y, Z, VolElementNo,
			  global_qdiv, global_Qdiv);

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
  
  time (&time_end);
  timeused = difftime (time_end,time_start);
  cout << " time used up (S) = " << timeused << "sec." << endl;
  
  /*
  delete[] T_Vol;
  delete[] kl_Vol;   
  delete[] scatter_Vol;
  delete[] rayNo_Vol;
  delete[] IntenArray_Vol;
  delete[] a_Vol;
  delete[] netInten_Vol;
  
  delete[] rayNo_top_surface;
  delete[] rayNo_bottom_surface;
  delete[] rayNo_front_surface;
  delete[] rayNo_back_surface;
  delete[] rayNo_left_surface;
  delete[] rayNo_right_surface;

  delete[] T_top_surface;
  delete[] T_bottom_surface;
  delete[] T_front_surface;
  delete[] T_back_surface;
  delete[] T_left_surface;
  delete[] T_right_surface;

  delete[] alpha_top_surface;
  delete[] alpha_bottom_surface;
  delete[] alpha_front_surface;
  delete[] alpha_back_surface;
  delete[] alpha_left_surface;
  delete[] alpha_right_surface;

  delete[] rs_top_surface;
  delete[] rs_bottom_surface;
  delete[] rs_front_surface;
  delete[] rs_back_surface;
  delete[] rs_left_surface;
  delete[] rs_right_surface;

  delete[] rd_top_surface;
  delete[] rd_bottom_surface;
  delete[] rd_front_surface;
  delete[] rd_back_surface;
  delete[] rd_left_surface;
  delete[] rd_right_surface;

  delete[] emiss_top_surface;
  delete[] emiss_bottom_surface;
  delete[] emiss_front_surface;
  delete[] emiss_back_surface;
  delete[] emiss_left_surface;
  delete[] emiss_right_surface;

  delete[] IntenArray_top_surface;
  delete[] IntenArray_bottom_surface;
  delete[] IntenArray_front_surface;
  delete[] IntenArray_back_surface;
  delete[] IntenArray_left_surface;
  delete[] IntenArray_right_surface;

  delete[] a_top_surface;
  delete[] a_bottom_surface;
  delete[] a_front_surface;
  delete[] a_back_surface;
  delete[] a_left_surface;
  delete[] a_right_surface;

  delete[] netInten_top_surface;
  delete[] netInten_bottom_surface;
  delete[] netInten_front_surface;
  delete[] netInten_back_surface;
  delete[] netInten_left_surface;
  delete[] netInten_right_surface;
  
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

  */
  return 0;


}

