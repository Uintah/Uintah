#include "ray.h"
#include "RNG.h"
#include "VirtualSurface.h"
#include "Consts.h"
#include "RealSurface.h"
#include "TopRealSurface.h"
#include "BottomRealSurface.h"
#include "FrontRealSurface.h"
#include "BackRealSurface.h"
#include "LeftRealSurface.h"
#include "RightRealSurface.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

class TopRealSurface;
class BottomRealSurface;
class FrontRealSurface;
class BackRealSurface;
class LeftRealSurface;
class RightRealSurface;

// for static data members, have to allocate the memory out of the class this way

double ray::pi = 3.1415926;
double ray::SB = 5.669 * pow(10., -8);


// get NoMedia from main function
ray::ray(const int &_BottomStartNo,
	 const int &_FrontStartNo,
	 const int &_BackStartNo,
	 const int &_LeftStartNo,
	 const int &_RightStartNo,
	 const int &_sumElementNo,
	 const int &_totalElementNo,
	 const int &_VolElementNo,
	 const int &_LeftRightNo,
	 const double &xc, const double &yc, const double &zc){
  
  BottomStartNo = _BottomStartNo;
  FrontStartNo = _FrontStartNo;
  BackStartNo = _BackStartNo;
  LeftStartNo = _LeftStartNo;
  RightStartNo = _RightStartNo;
  sumElementNo = _sumElementNo;
  totalElementNo = _totalElementNo;
  VolElementNo = _VolElementNo;
  LeftRightNo = _LeftRightNo;
  X = xc;
  Y = yc;
  Z = zc;
  
  
}

ray::~ray(){
  // delete obReal; // How to delete an object?? Is this right? 
}



int ray::get_currentIndex(){
  return currentIndex;
}

int ray::get_currentvIndex(){
  return currentvIndex;
}

// In this way, when T is not uniform within the element, calculate Emissive Energy for
// both the emissive point and the hit point

// this is for non-uniform T in each control element.
// may need the xhit, yhit, zhit; xemiss, yemiss, zemiss.



// to get q on surface elements ( make it efficient to calculate intensity
// on surface element 
double ray::SurfaceEmissFlux(const int &offset_index,
			     const double *emiss_surface,
			     const double *T_surface,
			     const double *a_surface){
 
  double emiss, Ts;

  // note:: this index might change as T is no longer in the array
  // *** the emiss might be different on different directions
  

  emiss = emiss_surface[offset_index];
  
  // T of surfaces can be a function of position too
  // actually, for non-uniform T, T is a function of position
  // ( xemiss, yemiss,zemiss)
  
 
  Ts = T_surface[offset_index] * T_surface[offset_index];
  
  // should use the black emissive intensity?
  // guess not the black one for surface? right?
  // cuz the attenuation caused by the medium didnot include
  // the surface absorption
  
  SurEmissFlux = emiss * SB * Ts * Ts * a_surface[offset_index];
  
  return SurEmissFlux;
  
}



double ray::SurfaceEmissFluxBlack(const int &offset_index,
				  const double *T_surface,
				  const double *a_surface){
 

  double Ts;

  // note:: this index might change as T is no longer in the array
  // *** the emiss might be different on different directions
  
 
  // emiss = emiss_surface[offset_index];
  

  Ts = T_surface[offset_index] * T_surface[offset_index];
  
  // should use the black emissive intensity?
  // guess not the black one for surface? right?
  // cuz the attenuation caused by the medium didnot include
  // the surface absorption
  
  SurEmissFlux =  SB * Ts * Ts * a_surface[offset_index]; 
  
  return SurEmissFlux;
  
}




// whether put the SurfaceEmissFlux inside the intensity function is a question
// 1. if just calculate q once, then to get different intensity , separate
// 2. acutally both q and I are just calculated once. put inside.

// to get Intensity on surface elements
double ray::SurfaceIntensity(const int &offset_index,
			     const double *emiss_surface,
			     const double *T_surface,
			     const double *a_surface){


  SurEmissFlux = SurfaceEmissFlux(offset_index, emiss_surface, T_surface, a_surface);
  
  // the integration over a surface is pi
  SurInten = SurEmissFlux / pi;
  
  return SurInten;
  
}


double ray::SurfaceIntensityBlack(const int &offset_index,
				  const double *T_surface,
				  const double *a_surface){


  SurEmissFlux =
    SurfaceEmissFluxBlack(offset_index, T_surface, a_surface);
  
  // the integration over a surface is pi
  SurInten = SurEmissFlux / pi;
  
  return SurInten;
  
}




double ray::VolumeEmissFluxBlack(const int &vIndex,
				 const double *T_Vol,
				 const double *a_Vol){
 

  // we need either the emission position or hit position
  
  double Ts;
  

  // T = f ( xemiss, yemiss, zemiss);
  //T = T0 * ( ( 1 - 2 * abs(xemiss)/X ) * ( 1 - 2 * abs(yemiss)/ Y) *
  //   ( 1 - 2 * abs(zemiss)/Z ) + 1 ); 

  // SB is the private static data member of class ray
  // set at the beginning of the code:

  
  Ts = T_Vol[vIndex] * T_Vol[vIndex];

  // black intensity of the medium??
  // cuz this scheme, the attenuation is already included in the path
  // like self-absorption

  VolEmissFlux = 4 * SB * Ts * Ts * a_Vol[vIndex];

  return VolEmissFlux;
    
}




double ray::VolumeEmissFlux(const int &vIndex,
			    const double *kl_Vol,
			    const double *T_Vol,
			    const double *a_Vol){


  // we need either the emission position or hit position
  
  double kl, T, Ts;
  

  kl = kl_Vol[vIndex];

  
  // T = f ( xemiss, yemiss, zemiss);
  //T = T0 * ( ( 1 - 2 * abs(xemiss)/X ) * ( 1 - 2 * abs(yemiss)/ Y) *
  //   ( 1 - 2 * abs(zemiss)/Z ) + 1 ); 

  // SB is the private static data member of class ray
  // set at the beginning of the code:
  

  T = T_Vol[vIndex];


  Ts = T * T;


  VolEmissFlux = 4 * kl * SB * Ts * Ts * a_Vol[vIndex];
  

  return VolEmissFlux;
   
}




double ray::VolumeIntensityBlack(const int &vIndex,
				 const double *T_Vol,
				 const double *a_Vol){


  VolEmissFlux = VolumeEmissFluxBlack(vIndex, T_Vol, a_Vol);

  
  // integrate over the whole medium, it is 4 * pi
  
  VolInten = VolEmissFlux / 4 / pi;
  
  return VolInten;
  
}




double ray::VolumeIntensity(const int &vIndex,
			    const double *kl_Vol,
			    const double *T_Vol,
			    const double *a_Vol){


  VolEmissFlux = VolumeEmissFlux(vIndex, kl_Vol, T_Vol, a_Vol);


  
  // integrate over the whole medium, it is 4 * pi
  VolInten = VolEmissFlux / 4 / pi;
  
  return VolInten;
  
}



  

int ray::get_emissSurfaceIndex(){
  return emissSurfaceIndex;
}

int ray::get_hitSurfaceIndex(){
  return hitSurfaceIndex;
}

void ray::get_emiss_point(double *emissP) const{
  emissP[0] = xemiss;
  emissP[1] = yemiss;
  emissP[2] = zemiss;
}

void ray::setEmissionPosition(){
  xemiss = xhit;
  yemiss = yhit;
  zemiss = zhit;
}

double ray::dotProduct(double *s1, const double *s2){
  return s1[0] * s2[0] + s1[1] * s2[1] + s1[2] * s2[2];
}

void ray::get_directionS(double *s){
  for ( int i = 0; i < 3; i ++ )
    directionVector[i] = s[i];
}

void ray::set_currentvIndex(const int &vIndex){
  currentvIndex = vIndex;
}

void ray::set_currentIndex(const int &index){
  currentIndex = index;
}


void ray::get_hit_point(double *hitP) const{
  hitP[0] = xhit;
  hitP[1] = yhit;
  hitP[2] = zhit;
}

double ray::ray_length(){
  length = sqrt ( ( xhit - xemiss ) * ( xhit - xemiss ) +
    ( yhit - yemiss ) * ( yhit - yemiss ) +
    ( zhit - zemiss ) * ( zhit - zemiss ) );
  return length;
}


void ray::get_specular_s(double *spec_s){
  
  double sum;
  sum = dotProduct(directionVector, surface_n[surfaceFlag]);
  for ( int i = 0; i < 3; i ++ ) { 
    spec_s[i] = directionVector[i] - 2 * sum * ( * ( surface_n[surfaceFlag] + i ) );
  }
    
}



// emission from control volume ( media )
void ray::getEmissionPositionVol( const double &xlow, const double &xup,
				  const double &ylow, const double &yup,
				  const double &zlow, const double &zup,
				  const int &vIndex){
  
  double random1, random2, random3;
  rng.RandomNumberGen(random1);
  rng.RandomNumberGen(random2);
  rng.RandomNumberGen(random3);
  xemiss = xlow + ( xup - xlow ) * random1;
  yemiss = ylow + ( yup - ylow ) * random2;
  zemiss = zlow + ( zup - zlow ) * random3;
  emissVolIndex = vIndex;
    
}

void ray::get_EmissSVol(double *sVol){
  double randomPhi, randomTheta;
  rng.RandomNumberGen(randomPhi);
  rng.RandomNumberGen(randomTheta);
  double phi, theta;
  phi = 2 * pi * randomPhi;
  theta = acos( 1 - 2 * randomTheta); 
  sVol[0] = sin(theta) * cos( phi ); // i 
  sVol[1] = sin( theta ) * sin ( phi ) ;// j 
  sVol[2] = 1 - 2 * randomTheta; // cos(theta), k 
}


// emission from real surfaces
void ray::getEmissionPosition( const double &alow,
			       const double &aup,
			       const double &blow,
			       const double &bup,
			       const double &constv,
			       const int &surfaceIndex){


  double random1, random2;
  rng.RandomNumberGen(random1);
  rng.RandomNumberGen(random2);

  // must be in this order, that surfaceIndex increases
  // different cases is cuz constv to be set to which

  //  cout << " ray getEmissionPosition line 39 " << endl;
  
  if (  surfaceIndex <  BottomStartNo ) { // top surface
    zemiss =  constv;
    xemiss =  alow + (  aup -  alow ) * random1;
    yemiss =  blow + (  bup -  blow ) * random2;
  }
  else if (  surfaceIndex <  FrontStartNo ) { // bottom surface
    zemiss =  constv;
    xemiss =  alow + (  aup -  alow ) * random1;
    yemiss =  blow + (  bup -  blow ) * random2;
  }
  else if (  surfaceIndex <  BackStartNo ) { // front surface
    yemiss =  constv;
    xemiss =  alow + (  aup -  alow ) * random1;
    zemiss =  blow + (  bup -  blow ) * random2;
  }
  else if (  surfaceIndex <  LeftStartNo ) { // back surface
    yemiss =  constv;
    xemiss =  alow + (  aup -  alow ) * random1;
    zemiss =  blow + (  bup -  blow ) * random2;
  }
  else if (  surfaceIndex <  RightStartNo) { // left surface
    xemiss =  constv;
    yemiss =  alow + (  aup -  alow ) * random1;
    zemiss =  blow + (  bup -  blow ) * random2;
  }
  else if (  surfaceIndex <  RightStartNo +  LeftRightNo ){ // right surface
    xemiss =  constv;
    yemiss =  alow + (  aup -  alow ) * random1;
    zemiss =  blow + (  bup -  blow ) * random2;
  }

  emissSurfaceIndex =  surfaceIndex;

}


// void ray::getEmissionPosition( RealSurface  *obReal, RNG &rng ){

//   double random1, random2;
//   rng.RandomNumberGen(random1);
//   rng.RandomNumberGen(random2);

//   if ( obReal->surfaceIndex < obReal->BottomStartNo ) { // top surface
//     zemiss = obReal->constv;
//     xemiss = obReal->alow + ( obReal->aup - obReal->alow ) * random1;
//     yemiss = obReal->blow + ( obReal->bup - obReal->blow ) * random2;
//   }
//   else if ( obReal->surfaceIndex < obReal->FrontStartNo ) { // bottom surface
//     zemiss = obReal->constv;
//     xemiss = obReal->alow + ( obReal->aup - obReal->alow ) * random1;
//     yemiss = obReal->blow + ( obReal->bup - obReal->blow ) * random2;
//   }
//   else if ( obReal->surfaceIndex < obReal->BackStartNo ) { // front surface
//     yemiss = obReal->constv;
//     xemiss = obReal->alow + ( obReal->aup - obReal->alow ) * random1;
//     zemiss = obReal->blow + ( obReal->bup - obReal->blow ) * random2;
//   }
//   else if ( obReal->surfaceIndex < obReal->LeftStartNo ) { // back surface
//     yemiss = obReal->constv;
//     xemiss = obReal->alow + ( obReal->aup - obReal->alow ) * random1;
//     zemiss = obReal->blow + ( obReal->bup - obReal->blow ) * random2;
//   }
//   else if ( obReal->surfaceIndex < obReal->RightStartNo) { // left surface
//     xemiss = obReal->constv;
//     yemiss = obReal->alow + ( obReal->aup - obReal->alow ) * random1;
//     zemiss = obReal->blow + ( obReal->bup - obReal->blow ) * random2;
//   }
//   else if ( obReal->surfaceIndex < obReal->RightStartNo + obReal->LeftRightNo ){ // right surface
//     xemiss = obReal->constv;
//     yemiss = obReal->alow + ( obReal->aup - obReal->alow ) * random1;
//     zemiss = obReal->blow + ( obReal->bup - obReal->blow ) * random2;
//   }

//   emissSurfaceIndex = obReal->surfaceIndex;

// }


// given VolIndex, find which surface of this cube the ray hits
// didnot change the currentvIndex, but to find which surface the ray hits on
double ray::surfaceIntersect(const double *VolTable ){

  double xcheck, ycheck, zcheck;
  double xlow, xup, ylow, yup, zlow, zup;
  double cc, index;
  
  const double *p; // non const pointer, const object

  // if dotProduct < 0 , have an intersection,
  // calculate intersect point
  
  // get volume top z, and xlimits, ylimits
  
  if ( dotProduct(directionVector, n_top) < 0 ) {
    
    p = VolTable;
    index = * ( p + 6 );
  
    //    cout << " might intersect with top surface " << endl;
    if ( index != currentvIndex ) {
      do {
	p += 13;
	index = * ( p + 6 );
      } while (index != currentvIndex);
    }
    if ( index == currentvIndex ) {
      zcheck = * p;
      yup = * ( p + 2 );
      ylow = * ( p + 3);
      xlow = * ( p + 4 );
      xup = * ( p + 5 );

      cc = ( zcheck - zemiss ) / directionVector[2];
      xcheck = directionVector[0] * cc + xemiss;
      ycheck = directionVector[1] * cc + yemiss;

//       cout << " xcheck = " << xcheck << endl;
//       cout << " ycheck = " << ycheck << endl;
//       cout << " zcheck = " << zcheck << endl;
      
      // check for limits
      if ( xcheck <= xup && xcheck >= xlow && ycheck <= yup && ycheck >= ylow ) {
	surfaceFlag = TOP;
	xhit = xcheck;
	yhit = ycheck;
	zhit = zcheck;
	hitSurfaceIndex = ( int ) * ( p + 7 );
	//	emissRayCounter[currentvIndex] ++;
	
	//	cout << " hitSurfaceIndex = " << hitSurfaceIndex << endl;
	//	cout << " ======== did hit on Top surface ==========" << endl;
	
	return 1;
      }
      
      else {
	//	cout << " not in range of top surface " << endl;

      }
      
    } //  else if
  } // dotProduct(s, n_top) < 0

  
  if ( dotProduct(directionVector, n_bottom) < 0 ) {
    p = VolTable;
    index = * ( p + 6 );
  
    //   cout << " might intersect with bottom surface " << endl;
    if ( index != currentvIndex ) {
      do {
	p += 13;
	index = * ( p + 6 );
      } while (index != currentvIndex);
    }
    if ( index == currentvIndex ) {
      //      cout << " if hit on bottom --- index = " << index << endl;
      zcheck = * ( p + 1 );
      yup = * ( p + 2 );
      ylow = * ( p + 3);
      xlow = * ( p + 4 );
      xup = * ( p + 5 );

      cc = ( zcheck - zemiss ) / directionVector[2];
      xcheck = directionVector[0] * cc + xemiss;
      ycheck = directionVector[1] * cc + yemiss;

//       cout << " xcheck = " << xcheck << endl;
//       cout << " ycheck = " << ycheck << endl;
//       cout << " zcheck = " << zcheck << endl;
      
      // check for limits
      if ( xcheck <= xup && xcheck >= xlow && ycheck <= yup && ycheck >= ylow ) {
	surfaceFlag = BOTTOM;
	xhit = xcheck;
	yhit = ycheck;
	zhit = zcheck;
	hitSurfaceIndex = ( int ) * ( p + 8 );
	//	emissRayCounter[currentvIndex] ++;
	
	//	cout << "hitSurfaceIndex =  " << hitSurfaceIndex << endl;
	//	cout << " ==================== did hit on Bottom Surface ==== " << endl;
	
	return 1;
      }
      else {
	//	cout << " not in range of bottom surface " << endl;

      }
      
    } //  else if
  } // dotProduct(s, n_bottom) < 0

  
  if ( dotProduct(directionVector, n_front) < 0 ) {
    p = VolTable;
    index = * ( p + 6 );
  
    //    cout << " might intersect with front surface " << endl;
    if ( index != currentvIndex ) {
      do {
	p += 13;
	index = * ( p + 6 );
      } while (index != currentvIndex);
    }
    if ( index == currentvIndex ) {
      ycheck = * ( p + 3);
      zup = * ( p );
      zlow = * ( p + 1);
      xlow = * ( p + 4 );
      xup = * ( p + 5 );

      cc = ( ycheck - yemiss ) / directionVector[1];
      xcheck = directionVector[0] * cc + xemiss;
      zcheck = directionVector[2] * cc + zemiss;

//       cout << " xcheck = " << xcheck << endl;
//       cout << " ycheck = " << ycheck << endl;
//       cout << " zcheck = " << zcheck << endl;
      
      // check for limits
      if ( xcheck <= xup && xcheck >= xlow && zcheck <= zup && zcheck >= zlow ) {
	surfaceFlag = FRONT;
	xhit = xcheck;
	yhit = ycheck;
	zhit = zcheck;
	hitSurfaceIndex = ( int ) * ( p + 9 );
	//	emissRayCounter[currentvIndex] ++;
	
	//	cout << "hitSurfaceIndex = " << hitSurfaceIndex << endl;
	//	cout << " ===== did hit on Front Surface ===== " << endl;
	
	return 1;
      }
      else {
	//	cout << " not in range of front surface " << endl;
      }
      
    } //  else if
  } // dotProduct(s, n_front) < 0


  if ( dotProduct(directionVector, n_back) < 0 ) {

    p = VolTable;
    index = * ( p + 6 );

    //    cout << " might intersect with back surface " << endl;
    if ( index != currentvIndex ) {
      do {
	p += 13;
	index = * ( p + 6 );
      } while (index != currentvIndex);
    }
    if ( index == currentvIndex ) {
      ycheck = * ( p + 2 );
      zup = * ( p );
      zlow = * ( p + 1);
      xlow = * ( p + 4 );
      xup = * ( p + 5 );

      cc = ( ycheck - yemiss ) / directionVector[1];
      xcheck = directionVector[0] * cc + xemiss;
      zcheck = directionVector[2] * cc + zemiss;

//       cout << " xcheck = " << xcheck << endl;
//       cout << " ycheck = " << ycheck << endl;
//       cout << " zcheck = " << zcheck << endl;
      
      // check for limits
      if ( xcheck <= xup && xcheck >= xlow && zcheck <= zup && zcheck >= zlow ) {
	surfaceFlag = BACK;
	xhit = xcheck;
	yhit = ycheck;
	zhit = zcheck;
	hitSurfaceIndex = ( int ) * ( p + 10 );
	//	emissRayCounter[currentvIndex] ++;
	
	//	cout << " hitSurfaceIndex = " << hitSurfaceIndex << endl;
	//	cout << " ===== did hit on Back Surface ==== " << endl;
	
	return 1;
      }
      else {
	//	cout << " not in range of back surface " << endl;
      }

    } //  else if
  } // dotProduct(s, n_back) < 0

  
  if ( dotProduct(directionVector, n_left) < 0 ) {
    
    p = VolTable;
    index = * ( p + 6 );
  
    //    cout << " might intersect with left surface " << endl;
    if ( index != currentvIndex ) {
      do {
	p += 13;
	index = * ( p + 6 );
      } while (index != currentvIndex);
    }
    if ( index == currentvIndex ) {
      xcheck = * ( p + 4 );
      yup = * ( p + 2 );
      ylow = * ( p + 3);
      zup = * ( p );
      zlow = * ( p + 1 );

      cc = ( xcheck - xemiss ) / directionVector[0];
      zcheck = directionVector[2] * cc + zemiss;
      ycheck = directionVector[1] * cc + yemiss;

//       cout << " xcheck = " << xcheck << endl;
//       cout << " ycheck = " << ycheck << endl;
//       cout << " zcheck = " << zcheck << endl;
      
      // check for limits
      if ( zcheck <= zup && zcheck >= zlow && ycheck <= yup && ycheck >= ylow ) {
	surfaceFlag = LEFT;
	xhit = xcheck;
	yhit = ycheck;
	zhit = zcheck;
	hitSurfaceIndex = ( int ) * ( p + 11 );
	
	//	cout << " hitSurfaceIndex = " << hitSurfaceIndex << endl;
	//	cout << " ===== did hit on Left surface ===== " << endl;
	
	return 1;
      }
      else {
	//	cout << " not in range of left surface " << endl;
      }
	
    } //  else if
  } // dotProduct(s, n_left) < 0
    
  
  if ( dotProduct(directionVector, n_right) < 0 ) {
    
    p = VolTable;
    index = * ( p + 6 );
  
    //    cout << " might intersect with right surface " << endl;
    if ( index != currentvIndex ) {
      do {
	p += 13;
	index = * ( p + 6 );
      } while (index != currentvIndex);
    }
    if ( index == currentvIndex ) {
      xcheck = * ( p + 5 );
      yup = * ( p + 2 );
      ylow = * ( p + 3);
      zup = * ( p );
      zlow = * ( p + 1 );

      cc = ( xcheck - xemiss ) / directionVector[0];
      zcheck = directionVector[2] * cc + zemiss;
      ycheck = directionVector[1] * cc + yemiss;
      
//       cout << " xcheck = " << xcheck << endl;
//       cout << " ycheck = " << ycheck << endl;
//       cout << " zcheck = " << zcheck << endl;
      
      // check for limits
      if ( zcheck <= zup && zcheck >= zlow && ycheck <= yup && ycheck >= ylow ) {
	surfaceFlag = RIGHT;
	xhit = xcheck;
	yhit = ycheck;
	zhit = zcheck;
	hitSurfaceIndex = ( int ) * ( p + 12 );
	//	emissRayCounter[currentvIndex] ++;
	
	//	cout << " hitSurfaceIndex = " << hitSurfaceIndex << endl;
	//	cout << " ===== Did hit on Right Surface ===== " << endl;
	
	return 1;
      }
      else {
	//	cout << " not in range of right surface " << endl;
      }
	
    } //  else if
  } // dotProduct(s, n_right) < 0      

  cout << " currentvIndex = " << currentvIndex << endl;
  cout << " xemiss = " << xemiss <<
    "; yemiss = " << yemiss <<
    "; zemiss = " << zemiss << endl;
  cout << "directionVector = " << directionVector[0] <<
    " i + " << directionVector[1] << " j + " <<
    directionVector[2] << "k " << endl;
  
  return 0; // return 0 after all these checks
}








// Backward Left Intensity

// surfaceFlag, currentvIndex, currentIndex, are the private data members of the class ray
// scattering and absorption in medium
// get the Er beforehand for two cases: uniform T, non-uniform T

// store path Index, Index's path length ( might not need to be stored),
// left fraction

void ray::TravelInMediumInten(const double *kl_Vol,
			      const double *scatter_Vol,
			      const double *VolNeighborArray,
			      const double *VolTableArray,
			      vector<double> &PathLeft,
			      vector<int> &PathIndex,
			      vector<double> &PathSurfaceLeft) {
  

  if ( ! surfaceIntersect(VolTableArray)) {
    cout << " didnot find the intersection surface" << endl;
    exit(1); // terminate the program
  }


  //surfaceIntersect(VolTableArray);
  length = ray_length();

  // calculate the energy lost during absorption in this currentvIndex
  double kl_m, scat_m;
  double random1, random2, random3, random;
  double sIncoming[3];
  double theta, phi;
  double LostF;    
  double scat_len; // scatter length
  double sumScat; // the sum of all the zigzag path within this subregion
  double extinc_medium;
  bool vIndexUpdate;
    
  vIndexUpdate = 0; // no update
  sumScat = 0;
  
  // look up vol property array for kl_m, and scat_m        
  kl_m = kl_Vol[currentvIndex];
  scat_m = scatter_Vol[currentvIndex];

  
  // automatically updated from the do loop
  // every next time to call this function, stores in the updated currentvIndex
  PathIndex.push_back(currentvIndex);
  
  // cout << " ***** in Travel function currentvIndex =  ****  "
  //    << currentvIndex << endl;
  
  // make sure the Index and the Left Portion is a match
//   cout << " ***** In TravelInMediumInten *** " << endl;
//   cout << "PathIndex : currentvIndex = " << currentvIndex << endl;


  do {
      
    rng.RandomNumberGen(random);

    // only use scattering coeff to get the scat_len.
    scat_len = - log(random) / scat_m;

 //    cout << "scat_len = " << scat_len << endl;
//     cout << " length = " << length << endl;
    
   
// 	cout << " xemiss = " << xemiss << endl;
// 	cout << " yemiss = " << yemiss << endl;
// 	cout << " zemiss = " << zemiss << endl;

// 	cout << " xhit = " << xhit << endl;
// 	cout << " yhit = " << yhit << endl;
// 	cout << " zhit = " << zhit << endl;
	
//  if ( length == 0 )
//       {
// 	exit (1);

//       }
    
    // no matter if scatter happens within this region, always use extinction coeff??
      
    if ( scat_len >= length ) { // within this subregion scatter doesnot happen
      
      sumScat = sumScat + length;
      
      if ( hitSurfaceIndex == -1 ) {// hit on virtual surface 
	// update vIndex as ray goes forward
	currentvIndex = ( int ) VolNeighborArray[ currentvIndex * 7 + surfaceFlag + 1];
	currentIndex = currentvIndex;
	PathSurfaceLeft.push_back(1);
      }
      else // hit on real surface

	{
	// the update for PathSurfaceLeft is in function hitRealSurfaceInten

	  currentIndex = hitSurfaceIndex;
	  //cout << "hitSurfaceIndex = " << hitSurfaceIndex << endl;

	}

      
      vIndexUpdate = 1;
      
      }
    else { // scatter happens within the subregion
      
      // scatter coeff + absorb coeff = scat_m + kl_m ---- so called the extinction coeff
      
      // update current position
      // x = xemiss + Dist * s[i];
      // y = yemiss + Dist * s[j];
	// z = zemiss + Dist * s[k];


      // cout << " scatter happens " << endl;
      
      xemiss = xemiss + scat_len * directionVector[0];
      yemiss = yemiss + scat_len * directionVector[1];
      zemiss = zemiss + scat_len * directionVector[2];
      sumScat = sumScat + scat_len;
      
      // set s be the sIncoming
      for ( int i = 0; i < 3 ; i ++ ) sIncoming[i] = directionVector[i];
      
      // get a new direction vector s
      obVirtual.get_s(rng, sIncoming, theta, phi, random1, random2, random3,
		      directionVector);
      
      // update on xhit, yhit, zhit, and ray_length
      if (surfaceIntersect(VolTableArray)) {
	length = ray_length();
      }
      else {
	cerr << " error @ not getting hit point coordinates after scattering! "
	     << endl;
	exit(1); // terminate program
      }      
    }
        
  } while ( ! vIndexUpdate );
  
  
  // update LeftFraction later


  // previous one
  //  extinc_medium =   (scat_m + kl_m) * sumScat ;

  // now consider in-scattering
  // for isotropic scattering
  extinc_medium =   (kl_m) * sumScat ;
 
  // store the a_n * delta l_n for each segement n
  PathLeft.push_back(extinc_medium);
  
 
  
}



// For Intensity

void ray::hitRealSurfaceInten(const double *absorb_surface,
			      const double *rs_surface,
			      const double *rd_surface,
			      vector<double> &PathSurfaceLeft){


  // for real surface,
  // PathLength stores the left percentage, 1 - alpha
  // PathIndex stores the index of the surface
  
  double alpha, rhos, rhod, ratio, random;
  double alpha_other, rhos_other, rhod_other;
  double spec_s[3];
  double diffuse_s[3];
  int offset = VolElementNo;
  int fake_hitSurfaceIndex = hitSurfaceIndex - offset;
  double LostF;

  // dealing with the surface element
  // calculate surface energy first, then push its index back in

  // already done in scatter_medium function
  //currentIndex = hitSurfaceIndex; // very important

  // check for absorption or reflection
  // look up surface property table to get absorption and reflection
  
  //alpha = RealSurfacePropertyArray[fake_hitSurfaceIndex * 6 + 2]; // absorption     
  //rhos = RealSurfacePropertyArray[fake_hitSurfaceIndex * 6 + 3]; // specular reflection
  //rhod = RealSurfacePropertyArray[fake_hitSurfaceIndex * 6 + 4]; // diffusive reflection


  alpha = absorb_surface[fake_hitSurfaceIndex];
  rhos = rs_surface[fake_hitSurfaceIndex];
  rhod = rd_surface[fake_hitSurfaceIndex];
  
  rng.RandomNumberGen( random );

 

//   cout << " **** In hitRealSurfaceInten **** " << endl;
//   cout << " PathIndex : hitSurfaceIndex = " << hitSurfaceIndex << endl;
  
  if ( alpha == 1 ) ratio = 10; // black body
  else ratio = rhod / ( rhos + rhod );
	 
 
  
  alpha_other = 1 - alpha;
  
  // the left part! the carrying on part
  PathSurfaceLeft.push_back(alpha_other);
 
  // check for the rest of the ray , if diffuse reflected or specular reflected
  if ( ratio <= 1   ) {
    
    if ( random <= ratio ) { // pure diffuse reflection
      
      // check which surface, top , bottom, front, back, left or right
      // must follow this order
      
      
      double theta, phi, random1, random2; // all these values obtained from get_s      
      if ( hitSurfaceIndex < BottomStartNo ) {// top surface
	TopRealSurface obReal;
	obReal.get_s(rng, theta, random1, phi, random2, diffuse_s);
      }
      else if ( hitSurfaceIndex < FrontStartNo ) {// bottom surface
	BottomRealSurface obReal;
	obReal.get_s(rng, theta, random1, phi, random2, diffuse_s);
      }
      else if ( hitSurfaceIndex < BackStartNo ) {// front surface
	FrontRealSurface obReal;
	obReal.get_s(rng, theta, random1, phi, random2, diffuse_s);
      }
      else if ( hitSurfaceIndex < LeftStartNo ) { // back surface
	BackRealSurface obReal;
	obReal.get_s(rng, theta, random1, phi, random2, diffuse_s);
      }
      else if ( hitSurfaceIndex < RightStartNo ) { // left surface
	LeftRealSurface obReal;
	obReal.get_s(rng, theta, random1, phi, random2, diffuse_s);
      }
      else if ( hitSurfaceIndex < sumElementNo ) {// right surface
	RightRealSurface obReal;
	obReal.get_s(rng, theta, random1, phi, random2, diffuse_s);
      }                  
      get_directionS(diffuse_s);
      
    }
    else { // pure specular reflection
      //    cout << " specular reflection-- part of energy lost " << endl;

      get_specular_s(spec_s);
      get_directionS(spec_s);
      
    }
    
  }

    
}  




void  ray::get_integrInten(double *integrInten,
			   const double *gg,
			   const double *netInten,
			   const int &ggNo, const int &local_ElementNo){
			   
  
  for ( int j = 0; j < local_ElementNo; j ++ ){
    
    for ( int i = 1; i < ggNo; i ++ ){

      integrInten[j] = integrInten[j] +
	( netInten[ i*local_ElementNo+j ] + netInten[ (i-1)*local_ElementNo+j] ) *
	(gg[i] - gg[i-1]) * 0.5;
      
    }

      
  }


  /*
  
  int flag_g = 0;
  int prevgIndex;
  int i = 1;
  
  do{
    
    if (( abs(gg[i] - gg[i-1] ) > 1e-10 ) && flag_g == 0  ){
      
      prevgIndex = i-1;
      
      for ( int j = 0; j < local_ElementNo; j ++ )
	integrInten[j] = integrInten[j] +
	  ( netInten[ i*local_ElementNo+j ] + netInten[ (prevgIndex)*local_ElementNo+j] ) *
	  (gg[i] - gg[prevgIndex]) * 0.5;
      
      i++;
    }   
    else if ( ( abs(gg[i] - gg[i-1] ) <= 1e-10 ) && flag_g == 0 ) { // gg[i] = gg[i-1]
      prevgIndex = i-1;
      flag_g = 1;
      i++;
    }
    else if ( ( abs(gg[i] - gg[i-1] ) <= 1e-10 ) && flag_g == 1 ){
      i++;
    }
    else if ( ( abs(gg[i] - gg[i-1] ) > 1e-10 ) && flag_g == 1 ) {
      
      for ( int j = 0; j < local_ElementNo; j ++ )
	integrInten[j] = integrInten[j] +
	  ( netInten[ i*local_ElementNo+j ] + netInten[ (prevgIndex)*local_ElementNo+j] ) *
	  (gg[i] - gg[prevgIndex]) * 0.5;
      
      flag_g = 0;
      i++;
      
    }
    
    
  }while( i<ggNo );
  
  */ 
    

}
  




  
