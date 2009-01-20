#include "MakeTableFunction.h"

#include <cmath>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <vector>
#include <sstream>

using namespace std;


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
  cout << "Nchalf = " << Nchalf << endl;
  cout << "Lhalf = " << Lhalf << endl;
  cout << "ratio = " << ratio << endl;
  cout << "pp = " << pp << endl;
  cout << "======================== " << endl;
  
  dcenter = (1-ratio)*Lhalf/(1- pow(ratio,Nchalf));    
  return dcenter;
}



int main(int argc, char *argv[]){

  int rayNoSurface, rayNoVol;
  double CubeLenx, CubeLeny, CubeLenz;
  double xc, yc, zc; // the length of on x, y, z coordinates
  int local_rayNoSurface, local_rayNoVol;
  char caseflag;
  double casePlates;
  int  mixtureKg, TablePoints;


  local_rayNoSurface = rayNoSurface;
  local_rayNoVol = rayNoVol;

  
  double pi = 3.1415926;
  double SB = 5.669 * pow(10., -8);
  

  // starting Nos on Top --- Right Surfaces
  int TopStartNo, BottomStartNo, FrontStartNo, BackStartNo, LeftStartNo, RightStartNo; 
  int VolElementNo, TopBottomNo, FrontBackNo, LeftRightNo, sumElementNo;
  //  int xno, yno, zno, xnop, ynop, znop;
  //  int size;
  int totalElementNo, surfaceElementNo;
  double EnergyAmount; // set as customer self-set-up later
  int surfaceIndex, currentIndex;
  double sumIncomInten, aveIncomInten;    
  



  // number of cells in each coordinates
  // make sure to pick the appropriate CubeLen to get integral xno, yno, zno


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
  double ratioBC;  // ratioBC = Boundary cell/ center cell
  double dxcenter, dycenter, dzcenter; // center cells' size

  cout << "please enter Ncx, Ncy, Ncz ( number of cells in x, y, z direction)" << endl;
  cin >> Ncx;
  cin >> Ncy;
  cin >> Ncz;

  cout << "Please enter Lx, Ly, Lz ( full length of full domain in x, y, z direction)" << endl;
  cin >> Lx;
  cin >> Ly;
  cin >> Lz;

  cout << "Please enter ratioBC = boundary cell/ center cell" << endl;
  cin >> ratioBC;
  
  
  Ncxhalf = Ncx / 2;
  Ncyhalf = Ncy / 2;
  Nczhalf = Ncz / 2;
  Lxhalf = Lx /2;
  Lyhalf = Ly/2;
  Lzhalf = Lz/2;
  
  // numbers of grid points
  Npx = Ncx + 1;
  Npy = Ncy + 1;
  Npz = Ncz + 1;
  
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
  
//   ray obRay(BottomStartNo, FrontStartNo,
// 	    BackStartNo, LeftStartNo, RightStartNo,
// 	    sumElementNo, totalElementNo, VolElementNo,
// 	    LeftRightNo, Lx, Ly, Lz);
  
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

  // get property of vol
  double *T_Vol = new double [VolElementNo];
  double *kl_Vol = new double [VolElementNo];
  double *scatter_Vol = new double [VolElementNo];
  double *a_Vol = new double[VolElementNo];
  int *rayNo = new int [totalElementNo];


  MakeTableFunction ob;

  // x direction
  int powNo;
  
  dxcenter = MeshSize(Ncxhalf,Lxhalf,ratioBC);
  cout << "dxcenter = " << dxcenter << endl;
  for ( int i = 0; i < Ncxhalf; i++ ){
    powNo = Ncxhalf-1-i;
    dx[i] = dxcenter * pow( ratioBC, powNo );
    cout << " dx[ " << i << "]= " << dx[i] << endl;
    dx[Ncx-i-1] = dx[i];
    cout << " dx[ " << Ncx-i-1 << "]= " << dx[Ncx-i-1] << endl;    
  }

  
  
  // y direction

  dycenter = MeshSize(Ncyhalf,Lyhalf,ratioBC);
  for ( int i = 0; i < Ncyhalf; i++ ) {
    dy[i] = dycenter * pow( ratioBC, Ncyhalf-1-i );
    dy[Ncy-i-1] = dy[i];
  }

  
  // z direction

  dzcenter = MeshSize(Nczhalf,Lzhalf,ratioBC);
  for ( int i = 0; i < Nczhalf; i++ ) {
    dz[i] = dzcenter * pow( ratioBC, Nczhalf-1-i );
    dz[Ncz-i-1] = dz[i];
  }

  
  // ========= set values or get values for array pointers =============
  // the center of the cube is at (0,0,0) in a cartesian coordinate
  // the orgin of the cube ( domain ) can be changed here easily
  
   X[0] = -Lx/2.; // start from left to right
   Y[0] = Ly/2.; // start from back to front
   Z[0] = Lz/2; // start from top to bottom
   X[Npx-1] = Lx/2;
   Y[Npy-1] = -Ly/2;
   Z[Npz-1] = -Lz/2;
   
   // dont use x[i] = f ( x[i-1] ) , will get fatal error when cubelen is not integer.

   for ( int i = 1; i < Ncxhalf ; i ++ )
     {
       X[i] = X[i-1] + dx[i-1];
       X[Ncx-i] = X[Npx-i] - dx[i-1];
       
     }

   ob.singleArrayTable(X,Npx,11,"Xtable");
   
   for ( int i = 1; i < Ncyhalf; i ++ )
     {
       Y[i] = Y[i-1] - dy[i-1];
       Y[Ncy-i] = Y[Npy-i] + dy[i-1];
     }

   ob.singleArrayTable(Y,Npy,11,"Ytable");

   
   for ( int i = 1; i < Nczhalf; i ++ )
     {
       Z[i] = Z[i-1] - dz[i-1];
       Z[Ncz-i] = Z[Npz-i] + dz[i-1];
     }

   ob.singleArrayTable(Z,Npz,11,"Ztable");

   
   X[Ncxhalf] = 0;
   Y[Ncxhalf] = 0;
   Z[Ncxhalf] = 0;   


   /*
     
   // Top and Bottom surfaces' surface element's area
   ElementAreaTB = CubeLenx * CubeLeny;
   
   // Front and Back surfaces' surface element's area
   ElementAreaFB = CubeLenx * CubeLenz;
   
   // Left and Right surfaces' surface element's area
   ElementAreaLR = CubeLeny * CubeLenz;
   
   double minArea;
   minArea = min(ElementAreaTB, ElementAreaFB);
   minArea = min(minArea, ElementAreaLR);
   
   
   // the ratio of Area with the minArea
   double ratioTB, ratioFB, ratioLR;
   
   ratioTB = ElementAreaTB / minArea;
   ratioFB = ElementAreaFB / minArea;
   ratioLR = ElementAreaLR / minArea;
   
   ElementVol = CubeLenx * CubeLeny * CubeLenz; // cell's volume
   
   
   for ( int i = 0; i < VolElementNo; i ++ )
   rayNo[i] = local_rayNoVol;
   
   for ( int i = VolElementNo; i < FrontStartNo; i ++ ) // top bottom surfaces
   rayNo[i] = int (local_rayNoSurface * ratioTB);
   
   for ( int i = FrontStartNo; i < LeftStartNo; i ++ ) // front back surfaces
   rayNo[i] = int (local_rayNoSurface * ratioFB);
   
   for ( int i = LeftStartNo; i < totalElementNo; i ++ ) // left right surfaces
     rayNo[i] = int (local_rayNoSurface * ratioLR);
  
   */

   
  // pick a quater of each surface and 1/8 of volume domain due to symmetry

  // Top and Bottom surfaces
   double minArea, minAreaTB, minAreaFB, minAreaLR;
   double minVol;
   
   double *ElementAreaTB = new double[TopBottomNo];
   double *ElementAreaFB = new double[FrontBackNo];
   double *ElementAreaLR = new double[LeftRightNo];
   double *ElementVol = new double[VolElementNo];


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


  
//    /*
//    double TBdim = Ncxhalf * Ncyhalf;
//    double *ElementAreaTB = new double[TBdim];

//    // Front and Back surfaces
//    double FBdim = Ncxhalf * Nczhalf;
//    double *ElementAreaFB = new double[FBdim];

//   // Left and Right surfaces
//    double LRdim = Ncyhalf * Nczhalf;
//   double *ElementAreaLR = new double[LRdim];


//   // volume domain
//   double Voldim = Ncxhalf * Ncyhalf * Nczhalf;
//   double *ElementVol = new double[Voldim];

  
//   // x direction first, then y direction
//   for ( int j = 0; j < Ncyhalf; j ++ )  
//     for ( int i = 0; i < Ncxhalf; i ++ )  
//       ElementAreaTB[ j*Ncxhalf + i ] = dx[i] * dy[j];


//   // x direction first, then z direction
//   for ( int j = 0; j < Nczhalf; j ++ )  
//     for ( int i = 0; i < Ncxhalf; i ++ )  
//       ElementAreaFB[ j*Ncxhalf + i ] = dx[i] * dz[j];
  

//   // y direction first, then z direction
//   for ( int j = 0; j < Nczhalf; j ++ )  
//     for ( int i = 0; i < Ncyhalf; i ++ )  
//       ElementAreaLR[ j*Ncyhalf + i ] = dy[i] * dz[j];
  
  
//   // element volume
//   // x direction, then y direction then z direction, so x-y layer by x-y layer
//   for ( int i = 0; i < Nczhalf; i ++ )
//     for ( int j = 0; j < Ncyhalf; j ++ )
//       for ( int k = 0; k < Ncxhalf; k ++ )
// 	ElementVol[ i*TBdim + j*Ncxhalf + k ] = dz[i] * dy[j] * dx[k];
//    */
   

  // setting rayNo for surfaces and volume depends on area and volume size
  // local_rayNoVol and local_rayNoSurface are for minArea and minVol
  
  if ( ratioBC < 1 ) // boundary has the finer mesh, so minArea are at the boundaries
    {
      minAreaTB = dx[0] * dy[0];
      minAreaFB = dx[0] * dz[0];
      minAreaLR = dy[0] * dz[0];
      minArea = min(minAreaTB, minAreaFB);
      minArea = min(minArea, minAreaLR);

      minVol = dx[0] * dy[0] * dz[0];

    }
  else if ( ratioBC > 1 ) // center has the finer mesh, so minArea are at the center
    {
      minAreaTB = dx[Ncxhalf-1] * dy[Ncyhalf-1];
      minAreaFB = dx[Ncxhalf-1] * dz[Nczhalf-1];
      minAreaLR = dy[Ncyhalf-1] * dz[Nczhalf-1];
      minArea = min(minAreaTB, minAreaFB);
      minArea = min(minArea, minAreaLR);

      minVol = dx[Ncxhalf-1] * dy[Ncyhalf-1] * dz[Nczhalf-1];

    }


  
  // assign rayNo for volume elements 
   for ( int i = 0; i < VolElementNo; i ++ )
     rayNo[i] = int (local_rayNoVol *  ElementVol[i] / minVol);
	  
        
   for ( int i = VolElementNo; i < FrontStartNo; i ++ ) // top bottom surfaces
     rayNo[i] = int (local_rayNoSurface * ElementAreaTB[i-VolElementNo] / minArea);

   for ( int i = FrontStartNo; i < LeftStartNo; i ++ ) // front back surfaces
     rayNo[i] = int (local_rayNoSurface * ElementAreaFB[i-FrontStartNo] / minArea);
 
   for ( int i = LeftStartNo; i < totalElementNo; i ++ ) // left right surfaces
     rayNo[i] = int (local_rayNoSurface * ElementAreaLR[i-LeftStartNo] / minArea);


    
   // only main processor generates tables and then arrays,
   // other processor will all copy data from the main arrays
   // these have to be defined globally.
   
   int VolTableSize;
   int VolNeighborSize;
   
   VolTableSize = VolElementNo * 13;
   double *VolTableArray = new double [VolTableSize];

   
   VolNeighborSize = VolElementNo * 7;
   double *VolNeighborArray = new double [VolNeighborSize];


   // to generate tables is for easy detecting errors
   // later on, will generate arrays directly on each processor

   ob.singleArrayTable(dx, Ncx, 10, "dxtable");
   ob.singleArrayTable(dy, Ncy, 10, "dytable");
   ob.singleArrayTable(dz, Ncz, 10, "dztable");
   
     //VolTable:
     // zup, zlow, yup, ylow, xlow, xup, vIndex, top, bottom, front, back, l, r   
     ob.VolTableMake(X, Y, Z,
		     Ncx, Ncy, Ncz,
		     TopStartNo,  BottomStartNo,
		     FrontStartNo, BackStartNo,
		     LeftStartNo,  RightStartNo,
		     TopBottomNo,  FrontBackNo, LeftRightNo,
		     "VolTableFlex");  
     ToArray(VolTableSize, VolTableArray, "VolTableFlex");
     
     ob.VolNeighbor(TopBottomNo, Ncx, Ncy, Ncz,
		    "VolNeighborTableFlex");
     ToArray(VolNeighborSize, VolNeighborArray, "VolNeighborTableFlex" );

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
     
  return 0;


}
  
