/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#include "ray.h"
#include "RNG.h"
#include "VirtualSurface.h"
#include "Consts.h"
#include "RealSurface.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;


// get NoMedia from main function
ray::ray(const int &_VolElementNo,
	 const int &Ncx_,
	 const int &Ncy_,
	 const int &Ncz_,
	 const int &offset_){
  
  
  VolElementNo = _VolElementNo;
  Ncx = Ncx_;
  Ncy = Ncy_;
  Ncz = Ncz_;
  offset = offset_;
  ghostX = Ncx + 2;
  ghostY = Ncy + 2;
  ghostTB = ghostX * ghostY;
}

ray::~ray(){
}



int ray::get_currentvIndex(){
  return currentvIndex;
}



int ray::get_hitSurfaceIndex(){
  return hitSurfaceIndex;
}


void ray::update_emissP(){
  xemiss = xhit;
  yemiss = yhit;
  zemiss = zhit;
}

double ray::dotProduct(const double *s1, const double *s2){
  return s1[0] * s2[0] + s1[1] * s2[1] + s1[2] * s2[2];
}


 void ray::get_directionS(double *s){
   for ( int i = 0; i < 3; i ++ )
     directionVector[i] = s[i];

 }


// inline
void ray::set_currentvIndex(const int &iIndex_,
			    const int &jIndex_,
			    const int &kIndex_){
   iIndex = iIndex_;
   jIndex = jIndex_;
   kIndex = kIndex_;
   
   currentvIndex = iIndex +
     jIndex * Ncx +
     kIndex * Ncx * Ncy;
   
 }


// inline
void ray::set_currentvIndex(const int &VolIndex){
  currentvIndex = VolIndex;
}


void ray::update_vIndex(){
  iIndex = futureViIndex;
  jIndex = futureVjIndex;
  kIndex = futureVkIndex;
  currentvIndex = iIndex +
     jIndex * Ncx +
     kIndex * Ncx * Ncy;
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



// emission from control volume ( media ) and surface
void ray::set_emissP( const double &xlow, const double &xup,
		      const double &ylow, const double &yup,
		      const double &zlow, const double &zup){
  
  double random1, random2, random3;
  rng.RandomNumberGen(random1);
  rng.RandomNumberGen(random2);
  rng.RandomNumberGen(random3);
  xemiss = xlow + ( xup - xlow ) * random1;
  yemiss = ylow + ( yup - ylow ) * random2;
  zemiss = zlow + ( zup - zlow ) * random3;
    
}

void ray::set_emissS_vol(double *sVol){
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



// void ray::set_emissP_surf(const double &xlow, const double &xup,
// 			  const double &ylow, const double &yup,
// 			  const double &zlow, const double &zup){
  
//   double random1, random2, random3;
//   rng.RandomNumberGen(random1);
//   rng.RandomNumberGen(random2);
//   rng.RandomNumberGen(random3);

//   xemiss = xlow + (xup - xlow) * random1;
//   yemiss = ylow + (yup - ylow) * random2;
//   zemiss = zlow + (zup - zlow) * random3;
// //   cout << "xemiss = " << xemiss << endl;
// //   cout << "yemiss = " << yemiss << endl;
// //   cout << "zemiss = " << zemiss << endl;
// }


double ray::get_xemiss(){
  return xemiss;
}

double ray::get_yemiss(){
  return yemiss;
}

double ray::get_zemiss(){
  return zemiss;
}



bool
ray::surfaceIntersect( const double *X,
                       const double *Y,
                       const double *Z,
                       const int *VolFeature ) {

  // s[0] = sin(theta) cos(phi)
  // s[1] = sin(theta) sin(phi)
  // s[2] = cos(theta)
  
  //  double xcheck, ycheck, zcheck;
  double xlow, xup, ylow, yup, zlow, zup;
  double cc;

  xx[0] = X[iIndex];
  xx[1] = X[iIndex+1];
  yy[0] = Y[jIndex];
  yy[1] = Y[jIndex+1];
  zz[0] = Z[kIndex];
  zz[1] = Z[kIndex+1];

  // if directionVector changes, then recalculate sign.
  if (dirChange) {
    for ( int i = 0; i < 3; i ++) {      
      // to inverse is necessary to check 1/-0.0 = - inf, and 1/0.0 = +inf    
      inv_directionVector[i] = 1/directionVector[i];
      sign[i] = (inv_directionVector[i] > 0);
    }
  }

  
  disX = ( xx[sign[0]] - xemiss ) * inv_directionVector[0];
  disY = ( yy[sign[1]] - yemiss ) * inv_directionVector[1];
  if ( disX <= disY ) {
    disMin = disX;
    surfaceFlag = sign[0] ? RIGHT : LEFT;
  }
  else
    {
      disMin = disY;
      surfaceFlag = sign[1] ? BACK : FRONT;
    }

  disZ = ( zz[sign[2]] - zemiss ) * inv_directionVector[2];

  if ( disZ <= disMin ) {
    disMin = disZ;
    
    surfaceFlag = sign[2] ? TOP : BOTTOM;
  }


    // could possible hit on top surface
  if ( surfaceFlag == TOP ) {

    xhit = directionVector[0] * disMin + xemiss;
    yhit = directionVector[1] * disMin + yemiss;
    zhit = Z[kIndex+1];
    

    // hitSurfaceIndex's position indices
    hitSurfaceiIndex = iIndex;
    hitSurfacejIndex = jIndex;
    hitSurfacekIndex = kIndex + 1;
    
    // jump ahead to get the direction of the ray when hit on top surface
    if (VolFeature[hitSurfaceiIndex +
		   hitSurfacejIndex * ghostX +
		   hitSurfacekIndex *ghostTB +
		   offset]){
      //	cout << "hit on top virtual" << endl;
      
      // update next step's volume index i, j, k, but note, not updating currentvIndex yet
      futureViIndex = hitSurfaceiIndex;
      futureVjIndex = hitSurfacejIndex;
      futureVkIndex = hitSurfacekIndex;
      
      // make sure that if not hit on realsurface and called hitSurfaceIndex
      // will return error
      hitSurfaceIndex = -1;	
      VIRTUAL =  1;
    }
    else{
      //	cout << "hit on top real " << endl;
      VIRTUAL = 0;
      futureViIndex = iIndex;
      futureVjIndex = jIndex;
      futureVkIndex = kIndex;	
      hitSurfaceIndex = hitSurfaceiIndex + hitSurfacejIndex * Ncx;
      obReal = &obTop_ray;
    }
    
    //       cout << "hitSurfaceiIndex = " << hitSurfaceiIndex << endl;
    //       cout << "hitSurfacejIndex = " << hitSurfacejIndex << endl;
    //       cout << "hitSurfacekIndex = " << hitSurfacekIndex << endl;
    return true;
    
       
  }
  
  
  // could possible hit on bottom surface
  if ( surfaceFlag == BOTTOM ) {
    
    xhit = directionVector[0] * disMin + xemiss;
    yhit = directionVector[1] * disMin + yemiss;
    zhit = Z[kIndex];

    // hitSurfaceIndex's position indices
    hitSurfaceiIndex = iIndex;
    hitSurfacejIndex = jIndex;
    hitSurfacekIndex = kIndex;
    
    // jump ahead to get the direction of the ray when hit on top surface
    if (VolFeature[hitSurfaceiIndex +
		   hitSurfacejIndex *ghostX +
		   (hitSurfacekIndex-1) *ghostTB +
		   offset]){
      //	cout << "hit on bottom virtual " << endl;
      
      futureViIndex = hitSurfaceiIndex;
      futureVjIndex = hitSurfacejIndex;
      futureVkIndex = hitSurfacekIndex-1;
      hitSurfaceIndex = -1;		
      VIRTUAL =  1;
    }
    else{
      //	cout << "hit on bottom real " << endl;
      futureViIndex = iIndex;
      futureVjIndex = jIndex;
      futureVkIndex = kIndex;	
      VIRTUAL = 0;
      hitSurfaceIndex = hitSurfaceiIndex + hitSurfacejIndex * Ncx;
      obReal = &obBottom_ray;
    }
    //       cout << "hitSurfaceiIndex = " << hitSurfaceiIndex << endl;
    //       cout << "hitSurfacejIndex = " << hitSurfacejIndex << endl;
    //       cout << "hitSurfacekIndex = " << hitSurfacekIndex << endl;      
    return true;
    

  }
  


  // could possible hit on front surface
  if ( surfaceFlag == FRONT ) {

    xhit = directionVector[0] * disMin + xemiss;
    yhit =  Y[jIndex];
    zhit = directionVector[2] * disMin + zemiss;
    
    // hitSurfaceIndex's position indices
    hitSurfaceiIndex = iIndex;
    hitSurfacejIndex = jIndex;
    hitSurfacekIndex = kIndex;
    
    // jump ahead to get the direction of the ray when hit on top surface
    if (VolFeature[hitSurfaceiIndex +
		   (hitSurfacejIndex-1) *ghostX +
		   hitSurfacekIndex *ghostTB +
		   offset]){
      //	cout << "hit on front virtual " << endl;
      
      futureViIndex = hitSurfaceiIndex;
      futureVjIndex = hitSurfacejIndex-1;
      futureVkIndex = hitSurfacekIndex;	
      hitSurfaceIndex = -1;	
      VIRTUAL =  1;
    }
    else{
      //	cout << "hit on front real " << endl;
      futureViIndex = iIndex;
      futureVjIndex = jIndex;
      futureVkIndex = kIndex;		
      VIRTUAL = 0;      
      hitSurfaceIndex = hitSurfaceiIndex + hitSurfacekIndex * Ncx;
      obReal = &obFront_ray;
    }
    //       cout << "hitSurfaceiIndex = " << hitSurfaceiIndex << endl;
    //       cout << "hitSurfacejIndex = " << hitSurfacejIndex << endl;
    //       cout << "hitSurfacekIndex = " << hitSurfacekIndex << endl;      
    return true;
    
 
  }



  // could possible hit on back surface
  if ( surfaceFlag == BACK ) {

    xhit = directionVector[0] * disMin + xemiss;
    yhit = Y[jIndex+1];
    zhit = directionVector[2] * disMin + zemiss;
    
    // hitSurfaceIndex's position indices
    hitSurfaceiIndex = iIndex;
    hitSurfacejIndex = jIndex+1;
    hitSurfacekIndex = kIndex;
    
    // jump ahead to get the direction of the ray when hit on top surface
    if (VolFeature[hitSurfaceiIndex +
		   hitSurfacejIndex *ghostX +
		   hitSurfacekIndex *ghostTB +
		   offset]){
      //	cout << "hit on back virtual " << endl;
      futureViIndex = hitSurfaceiIndex;
      futureVjIndex = hitSurfacejIndex;
      futureVkIndex = hitSurfacekIndex;
      hitSurfaceIndex = -1;		
      VIRTUAL =  1;
    }
    else{
      //	cout << "hit on back real " << endl;
      futureViIndex = iIndex;
      futureVjIndex = jIndex;
      futureVkIndex = kIndex;		
      VIRTUAL = 0;      
      hitSurfaceIndex = hitSurfaceiIndex + hitSurfacekIndex * Ncx;
      obReal = &obBack_ray;
    }
    //       cout << "hitSurfaceiIndex = " << hitSurfaceiIndex << endl;
    //       cout << "hitSurfacejIndex = " << hitSurfacejIndex << endl;
    //       cout << "hitSurfacekIndex = " << hitSurfacekIndex << endl;      
    return true;
    
  }
  
  
   
  // could possible hit on left surface
  if ( surfaceFlag == LEFT ) {

    xhit = X[iIndex];
    yhit = directionVector[1] * disMin + yemiss;
    zhit = directionVector[2] * disMin + zemiss;
    
    // hitSurfaceIndex's position indices
    hitSurfaceiIndex = iIndex;
    hitSurfacejIndex = jIndex;
    hitSurfacekIndex = kIndex;
    
    // jump ahead to get the direction of the ray when hit on top surface
    if (VolFeature[(hitSurfaceiIndex-1) +
		   hitSurfacejIndex *ghostX +
		   hitSurfacekIndex *ghostTB +
		   offset]){
      //	cout << " hit on left virtual " << endl;
      futureViIndex = hitSurfaceiIndex-1;
      futureVjIndex = hitSurfacejIndex;
      futureVkIndex = hitSurfacekIndex;
      hitSurfaceIndex = -1;		
      VIRTUAL =  1;
    }
    else{
      //	cout << "hit on left real " << endl;
      futureViIndex = iIndex;
      futureVjIndex = jIndex;
      futureVkIndex = kIndex;		
      VIRTUAL = 0;      
      hitSurfaceIndex = hitSurfacejIndex + hitSurfacekIndex * Ncy;
      obReal = &obLeft_ray;
    }
    //      cout << "hitSurfaceiIndex = " << hitSurfaceiIndex << endl;
    //       cout << "hitSurfacejIndex = " << hitSurfacejIndex << endl;
    //       cout << "hitSurfacekIndex = " << hitSurfacekIndex << endl;      
    return true;

  }
  


  
  // could possible hit on right surface
  if ( surfaceFlag == RIGHT ) {

    xhit = X[iIndex+1];
    yhit = directionVector[1] * disMin + yemiss;
    zhit = directionVector[2] * disMin + zemiss;
    
    // hitSurfaceIndex's position indices
    hitSurfaceiIndex = iIndex+1;
    hitSurfacejIndex = jIndex;
    hitSurfacekIndex = kIndex;
    
    // jump ahead to get the direction of the ray when hit on top surface
    if (VolFeature[hitSurfaceiIndex +
		   hitSurfacejIndex *ghostX +
		   hitSurfacekIndex *ghostTB +
		   offset]){
      //	cout << " hit on right virtual " << endl;
      futureViIndex = hitSurfaceiIndex;
      futureVjIndex = hitSurfacejIndex;
      futureVkIndex = hitSurfacekIndex;
      hitSurfaceIndex = -1;		
      VIRTUAL =  1;
    }
    else{
      //	cout << " hit on right real " << endl;
      futureViIndex = iIndex;
      futureVjIndex = jIndex;
      futureVkIndex = kIndex;		
      VIRTUAL = 0;      
      hitSurfaceIndex = hitSurfacejIndex + hitSurfacekIndex * Ncy;
      obReal = &obRight_ray;
    }
    //       cout << "hitSurfaceiIndex = " << hitSurfaceiIndex << endl;
    //       cout << "hitSurfacejIndex = " << hitSurfacejIndex << endl;
    //       cout << "hitSurfacekIndex = " << hitSurfacekIndex << endl;      
    return true;

  }


  cerr << " No surfaces hit!" << endl;
  return false; // no surface hit
  
} // end surfaceIntersect()








// Backward Left Intensity

// surfaceFlag, currentvIndex, currentIndex, are the private data members of the class ray
// scattering and absorption in medium
// get the Er beforehand for two cases: uniform T, non-uniform T

// store path Index, Index's path length ( might not need to be stored),
// left fraction

void ray::TravelInMediumInten(const double *kl_Vol,
			      const double *scatter_Vol,
			      const double *X,
			      const double *Y,
			      const double *Z,
			      const int *VolFeature,
			      double &PathLeft,
			      double &PathSurfaceLeft) {
  

  if ( !surfaceIntersect(X, Y, Z, VolFeature) ) {
    cout << "Error: did not find the intersection surface.\n";
    exit(1);
  }

  length = ray_length();
  //  cout << "length = " << length << endl;
  
  // calculate the energy lost during absorption in this currentvIndex
  double kl_m, scat_m;
  double random1, random2, random3, random;
  double sIncoming[3];
  double scat_len; // scatter length
  double sumScat; // the sum of all the zigzag path within this subregion
  bool vIndexUpdate;
    
  vIndexUpdate = 0; // no update
  sumScat = 0;
  
  // look up vol property array for kl_m, and scat_m
  // got the currentvIndex from ray.cc function get_currentvIndex()
  // and this is based on iIndex, jIndex, kIndex
  kl_m = kl_Vol[currentvIndex];
  scat_m = scatter_Vol[currentvIndex];
  dirChange = 0; 
   
  do {
      
    rng.RandomNumberGen(random);
    // the random number from drand48() could return [0,1)
    // to avoid log(zero), now we use log(1-random)
    
    // random number from Mersenee Twister can return either way,
    // here we use (0,1]
    
    // only use scattering coeff to get the scat_len.
    scat_len = - log(random) / scat_m;

    if ( scat_len >= length ) { // within this subregion scatter doesnot happen
      
      sumScat = sumScat + length;
      
      // no matter if it is virtual or real,
      // if real, the PathSurfaceLeft will be updated again later
      PathSurfaceLeft = 1;      
      vIndexUpdate = 1;
      
      }
    else { // scatter happens within the subregion
      
      // scatter coeff + absorb coeff = scat_m + kl_m ---- so called the extinction coeff
      
      // update current position
      // x = xemiss + Dist * s[i];
      // y = yemiss + Dist * s[j];
      // z = zemiss + Dist * s[k];
      dirChange = 1;
      xemiss = xemiss + scat_len * directionVector[0];
      yemiss = yemiss + scat_len * directionVector[1];
      zemiss = zemiss + scat_len * directionVector[2];
      sumScat = sumScat + scat_len;
      
      // set s be the sIncoming
       for ( int i = 0; i < 3 ; i ++ ) sIncoming[i] = directionVector[i];
      
      // get a new direction vector s
      obVirtual.get_s(rng, sIncoming, directionVector);
      
      // update on xhit, yhit, zhit, and ray_length
      if( surfaceIntersect(X, Y, Z, VolFeature) ) {
	length = ray_length();
      }
      else {
	cerr << " error @ not getting hit point coordinates after scattering!\n";
	exit(1); // terminate program
      }
      
    }// else scatter happens

    
  } while ( ! vIndexUpdate );
  
  
  // update LeftFraction later

  //  extinc_medium =   (kl_m) * sumScat ;
 
  // store the a_n * delta l_n for each segement n
  PathLeft =  (kl_m) * sumScat ;
  
}



// For Intensity
void ray::hitRealSurfaceInten(const double *absorb_surface,
			      const double *rs_surface,
			      const double *rd_surface,
			      double &PathSurfaceLeft){


  // for real surface,
  // PathLength stores the left percentage, 1 - alpha
  // PathIndex stores the index of the surface
  
  double alpha, rhos, rhod, ratio, random;
  double alpha_other, rhos_other, rhod_other;
  double spec_s[3];

  // dealing with the surface element
  // calculate surface energy first, then push its index back in

  // already done in scatter_medium function
  //  currentIndex = hitSurfaceIndex; // very important
  
  dirChange = 1;
  
  alpha = absorb_surface[hitSurfaceIndex];
  rhos = rs_surface[hitSurfaceIndex];
  rhod = rd_surface[hitSurfaceIndex];
  //  cout << "hitSurfaceIndex inside = " << hitSurfaceIndex << endl;
  rng.RandomNumberGen( random );

  if ( alpha == 1 ) ratio = 10; // black body
  else ratio = rhod / ( rhos + rhod );

  // the left part! the carrying on part
  PathSurfaceLeft = 1 - alpha;

  // check for the rest of the ray , if diffuse reflected or specular reflected
  if ( ratio <= 1   ) {
    // cout << "random for reflection = " << random << endl;
    if ( random <= ratio ) { // pure diffuse reflection
      // cout << "diffuse" << endl;
      // check which surface, top , bottom, front, back, left or right
      // must follow this order
      obReal->get_s(rng, directionVector);
      //      cout << "ray line 732 " << endl;
    }
    else { // pure specular reflection
      //  cout << "specular " << endl;
      //     cout << "ray line 735 " << endl;
      get_specular_s(spec_s);
      get_directionS(spec_s);
      
    }
    
  }

    
}  





  
