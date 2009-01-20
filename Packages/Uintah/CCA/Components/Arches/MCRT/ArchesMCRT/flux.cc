#include "flux.h"

#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;

double flux::pi = 3.1415926;
double flux::SB = 5.669 * pow( 10., -8);

flux::flux(const int &_surfaceElementNo, const int &_VolElementNo, const int &_TopStartNo,
	   const int & _BottomStartNo, const int &_FrontStartNo,
	   const int &_BackStartNo, const int &_LeftStartNo,
	   const int &_RightStartNo, const int &_sumElementNo,
	   const int &_TopBottomNo, const int &_FrontBackNo, const int &_totalElementNo){
  surfaceElementNo = _surfaceElementNo;
  VolElementNo = _VolElementNo;
  TopStartNo = _TopStartNo;
  BottomStartNo = _BottomStartNo;
  FrontStartNo = _FrontStartNo;
  BackStartNo = _BackStartNo;
  LeftStartNo = _LeftStartNo;
  RightStartNo = _RightStartNo;
  sumElementNo = _sumElementNo; 
  TopBottomNo = _TopBottomNo;
  FrontBackNo = _FrontBackNo;
  totalElementNo = _totalElementNo;
  
  // the sequence of index starts with control volume
  // Dont re-declare these arrays.
  
  Q_surface = new double [surfaceElementNo]; // surface elements Q
  q_surface = new double [surfaceElementNo]; // surface elements q
  Qdiv = new double [VolElementNo]; // control volume divergence of Q
  qdiv = new double [VolElementNo];
  
}

flux::~flux(){
  delete[] Q_surface;
  delete[] q_surface;
  delete[] Qdiv;
  delete[] qdiv;
}

/*
double flux::EmissSurface(const int &offset_index,
			  const double &ElementArea,
			  const double *RealSurfacePropertyArray) {
  double T, emiss, eE; // emissive energy

  T = RealSurfacePropertyArray[ offset_index * 6 + 1];
  emiss = RealSurfacePropertyArray[ offset_index * 6 + 5]; 
  eE = emiss * SB * pow( T, 4.0 ) * ElementArea; 

  return eE;
  
}


double flux::EmissVol(const int &vIndex,
		      const double &ElementVol,
		      const double *VolPropertyArray) {
  double T, kl, eE, eEdir;
  T = VolPropertyArray[ vIndex * 5 + 1];
  kl = VolPropertyArray[ vIndex * 5 + 2];
  eE = 4 * kl * SB * pow ( T, 4.0 ) * ElementVol; // total in all directions
  
  return eE;
  
}



void flux::surfaceflux_div(const double *Fij,
			   const double *RealSurfacePropertyArray,
			   const double *VolPropertyArray,
			   const double *emissRayCounter,
			   const double *SurfaceEmissiveEnergy,
			   const double *VolEmissiveEnergy,
			   const double &ElementAreaTB,
			   const double &ElementAreaFB,
			   const double &ElementAreaLR,
			   const double &ElementVol) {
  
  // Q_surface = Q_emiss - Q_absorb;
  // Q > 0 emit energy, Q < 0 absorb energy
  
  double EmissEnergy = 0;
  double AbsorbEnergy = 0;
  int m = 2 * TopBottomNo;
  int n = m + 2 * FrontBackNo;  
  int k;
  
  // whole surfaces
  for ( int i = 0; i < surfaceElementNo; i ++ ) {

    EmissEnergy = SurfaceEmissiveEnergy[i];

    // Absorbed Energy by this surface ( from other surfaces and media )
    
    AbsorbEnergy = 0;

    for ( int j = 0; j < VolElementNo; j ++ ) { // control volume
      
      k = j * totalElementNo + ( i + TopStartNo ) ;
      AbsorbEnergy = AbsorbEnergy + Fij[k] * VolEmissiveEnergy[j]; //* emissRayCounter[j];
      
    }

    for ( int j = VolElementNo; j < totalElementNo; j ++ ) { // surface element

      // energy emitted from j surface, and absorbed by "surfaceIndex" surface
      k = j * totalElementNo + ( i + TopStartNo );
      
      // TopStartNo = VolElementNo
      AbsorbEnergy = AbsorbEnergy + Fij[k] * SurfaceEmissiveEnergy[j-VolElementNo];  

    }

    Q_surface[i] = EmissEnergy - AbsorbEnergy;
    
    
    if ( i < m )
      q_surface[i] = Q_surface[i] / ElementAreaTB;
    else if ( i < n )
      q_surface[i] = Q_surface[i] / ElementAreaFB;
    else
      q_surface[i] = Q_surface[i] / ElementAreaLR;
    
  }
  
 
  // Qdiv and qdiv for control volumes
  
  for ( int i = 0; i < VolElementNo; i ++ ) {
    
    EmissEnergy = VolEmissiveEnergy[i] ; //* emissRayCounter[i];
    
    AbsorbEnergy = 0;
   
    for ( int j = 0; j < VolElementNo; j ++ ) { // control volume
      
      k = j * totalElementNo + i;
      AbsorbEnergy = AbsorbEnergy + Fij[k] * VolEmissiveEnergy[j]; // * emissRayCounter[j];
      
    }

    for ( int j = VolElementNo; j < totalElementNo; j ++ ) { // surface element

      k = j * totalElementNo + i;      
      AbsorbEnergy = AbsorbEnergy + Fij[k] * SurfaceEmissiveEnergy[j-VolElementNo];

    }
      
    Qdiv[i] = EmissEnergy - AbsorbEnergy;
    qdiv[i] = Qdiv[i] / ElementVol;

  }

  sumheat = 0;
  sumsurface = 0;
  sumQtop = 0;
  sumQbottom = 0;
  sumQfront = 0;
  sumQback = 0;
  sumQleft = 0;
  sumQright = 0;
  
  for ( int i = 0; i < VolElementNo; i ++ )
    sumheat = sumheat + Qdiv[i] ;

  cout << " sumheat in volumes = " << sumheat << endl;

  int i = 0;
  for ( ; i < BottomStartNo - VolElementNo; i ++ ) // top surface
    sumQtop = sumQtop + Q_surface[i];

  for ( ; i < FrontStartNo - VolElementNo; i ++ ) // bottom surface
    sumQbottom = sumQbottom + Q_surface[i];

  for ( ; i < BackStartNo - VolElementNo; i ++ ) // front surface
    sumQfront = sumQfront + Q_surface[i];

  for ( ; i < LeftStartNo - VolElementNo; i ++ ) // back surface
    sumQback = sumQback + Q_surface[i];

  for ( ; i < RightStartNo - VolElementNo; i ++ ) // left surface
    sumQleft = sumQleft + Q_surface[i];

  for ( ; i < sumElementNo - VolElementNo; i ++ ) // right surface
    sumQright = sumQright + Q_surface[i];
  
  sumsurface = sumQtop + sumQbottom + sumQfront +
    sumQback + sumQleft + sumQright;

  sumheat = sumheat + sumsurface;
  
  // the all sumup cannot garauntee that the results are right
  cout << " sumheat only on surfaces sumsurface = " << sumsurface << endl;
  
  cout << " sumheat in all domain ( including surfaces and volumes ) = " <<
    sumheat << endl;
  
}

void flux::norm_error(const double &topexact, const double &bottomexact,
		      const double &frontexact, const double &backexact,
		      const double &leftexact, const double &rightexact,
		      const double &xc, const double &yc, const double &zc){
  double errTop, errBottom, errFront, errBack, errLeft, errRight;
  double TBarea, FBarea, LRarea;
  TBarea = xc * yc;
  FBarea = xc * zc;
  LRarea = yc * zc;
  
  // calculating flux
  sumQtop = sumQtop / TBarea;
  sumQbottom = sumQbottom / TBarea;
  sumQfront = sumQfront / FBarea;
  sumQback = sumQback / FBarea;
  sumQleft = sumQleft / LRarea;
  sumQright = sumQright / LRarea;

  cout << " sumQleft = " << sumQleft << endl;
  cout << " sumQbottom = " << sumQbottom << endl;
  cout << " sumQfront = " << sumQfront << endl;
  cout << " sumQback = " << sumQback << endl;
  cout << " sumQtop = " << sumQtop << endl;
  cout << " sumQright = " << sumQright << endl;
  
  errTop = ( topexact - sumQtop ) / abs( topexact );
  errBottom = ( bottomexact - sumQbottom ) / abs( bottomexact );
  // errFront =  ( frontexact - sumQfront ) / abs( frontexact );
  //  errBack =  ( backexact - sumQback ) / abs( backexact );
    //  errLeft = ( leftexact - sumQleft ) / abs( leftexact );
  // errRight =  ( rightexact - sumQright ) / abs( rightexact );
  cout << " errTop = " << errTop << endl;   
  // cout << " errBack = " << errBack << endl;
  //cout << " errFront = " << errFront << endl;
  cout << " errBottom = " << errBottom << endl;
  // cout << " errLeft = " << errLeft << endl;
  //  cout << " errRight = " << errRight << endl;
 
}
		      
*/

void flux::net_div_flux(const double *EmissE, const double *AbsorbTerm,
			const double &ElementAreaTB, const double &ElementAreaFB,
			const double &ElementAreaLR, const double &ElementVol){

  int m = 2 * TopBottomNo;
  int n = m + 2 * FrontBackNo;
  double sumAllQ = 0;


  // surfaces
  int i;
  for ( i = 0; i < m; i ++ ) { // top and bottom surfaces
    Q_surface[i] = EmissE[i + VolElementNo] - AbsorbTerm[i + VolElementNo];
    q_surface[i] = Q_surface[i] / ElementAreaTB;
  }

  for ( ; i < n; i ++ ) { // front and back surfaces
    Q_surface[i] = EmissE[i + VolElementNo] - AbsorbTerm[i + VolElementNo];
    q_surface[i] = Q_surface[i] / ElementAreaFB;
  }

  for ( ; i < surfaceElementNo; i ++ ) { // left and right surfaces
    Q_surface[i] = EmissE[i + VolElementNo] - AbsorbTerm[i + VolElementNo];
    q_surface[i] = Q_surface[i] / ElementAreaLR;
  }

  
  // vol
  for ( int k = 0; k < VolElementNo; k ++ ) {
    Qdiv[k] = EmissE[k] - AbsorbTerm[k]; // no offset needed here
    qdiv[k] = Qdiv[k] / ElementVol;
  }

  for ( int k = 0 ; k < VolElementNo; k ++ ) 
    sumAllQ = sumAllQ + Qdiv[k];

  for ( int k = 0; k < surfaceElementNo; k ++ )
    sumAllQ = sumAllQ + Q_surface[k];

  cout << " sumAllQ = " << sumAllQ << endl;
  
}


  
