/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef ray_H
#define ray_H

#include <cstdlib>
#include <cmath>

#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/MersenneTwister.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/VirtualSurface.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/RealSurface.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/TopRealSurface.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/BottomRealSurface.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/FrontRealSurface.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/BackRealSurface.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/LeftRealSurface.h>
#include  <CCA/Components/Arches/MCRT/ArchesRMCRT/RightRealSurface.h>


class MTRand;
class VirtualSurface;
class RealSurface;
class TopRealSurface;
class BottomRealSurface;
class FrontRealSurface;
class BackRealSurface;
class LeftRealSurface;
class RightRealSurface;

using namespace std;

class ray{

public:

  ray(const int &_VolElementNo,
      const int &Ncx_, const int &Ncy_, const int &Ncz_);

  
  inline
  int get_hitSurfaceIndex(){
    return hitSurfaceIndex;
  }
  

  inline
  void set_emissP(MTRand &MTrng,
		  const double &xlow, const double &xup,
		  const double &ylow, const double &yup,
		  const double &zlow, const double &zup){

    R_xemiss =  MTrng.randExc();
    R_yemiss =  MTrng.randExc();
    R_zemiss =  MTrng.randExc();
    
    xemiss = xlow + ( xup - xlow ) * R_xemiss;
    yemiss = ylow + ( yup - ylow ) * R_yemiss;
    zemiss = zlow + ( zup - zlow ) * R_zemiss;
    
  }
  
  
 
  void set_emissS_vol(MTRand &MTrng,
		      double *sVol);
  
  
  // Travelling Ray's Emission Point
   // set hit point as next emission point,
  // and set hit point surface as new emission point surface

  inline
  void update_emissP(){
    xemiss = xhit;
    yemiss = yhit;
    zemiss = zhit;
  }
  
  // checking which surfaces the ray get intersection in
  // Returns true if a surface was hit.
  bool surfaceIntersect( const double *X,
                         const double *Y,
                         const double *Z,
                         const int *VolFeature );
 
  inline
  void set_directionS(const double *s){
    for ( int i = 0; i < 3; i ++ )
      directionVector[i] = s[i];    
  }
  

   inline
   void get_directionS(double *s){
     for ( int i = 0; i < 3; i ++ )
       s[i] = directionVector[i];
   }

  
  inline
  double ray_length(){
    length = sqrt ( ( xhit - xemiss ) * ( xhit - xemiss ) +
		    ( yhit - yemiss ) * ( yhit - yemiss ) +
		    ( zhit - zemiss ) * ( zhit - zemiss ) );
    return length;
  } 


  inline
  double get_ray_length(){
    return length;
  }

  
  inline
  int get_surfaceFlag()
  {
    return surfaceFlag;
  }
  

  inline
  void set_currentvIndex(const int &iIndex_,
			 const int &jIndex_,
			 const int &kIndex_){
    iIndex = iIndex_;
    jIndex = jIndex_;
    kIndex = kIndex_;
    
    currentvIndex = iIndex +
      jIndex * Ncx +
      kIndex * Ncx * Ncy;
    
    //  if (directionVector[2] >  0 ) currentvIndex = 32779; //4189; //3789;
    //   else currentvIndex = 31179;//3789; //4189;
     
  }
  

  inline
  void set_currentvIndex(const int &VolIndex){
    currentvIndex = VolIndex;
  }

  
  inline
  void set_futurevIndex(const int &iIndex_,
			const int &jIndex_,
			const int &kIndex_){
    
    futureViIndex = iIndex_;
    futureVjIndex = jIndex_;
    futureVkIndex = kIndex_;  
    
    futurevIndex = futureViIndex +
      futureVjIndex * Ncx +
      futureVkIndex * Ncx * Ncy;
    
  }


  inline
  int get_currentvIndex(){
    return currentvIndex;
  }
  

  inline
  int get_futurevIndex(){
    return futurevIndex;
  }

  
  inline
  void update_vIndex(){
    iIndex = futureViIndex;
    jIndex = futureVjIndex;
    kIndex = futureVkIndex;
    currentvIndex = futurevIndex;

  }
  

  inline
  double get_xemiss(){
    return xemiss;
  }
  
  inline
  double get_yemiss(){
    return yemiss;
  }


  inline
  double get_zemiss(){
    return zemiss;
  }

  inline
  double get_R_xemiss(){
    return R_xemiss;
  }

  
  inline
  double get_R_yemiss(){
    return R_yemiss;
  }


  inline
  double get_R_zemiss(){
    return R_zemiss;
  }

  
  inline
  double get_R_theta(){
    return R_theta;
  }


//   inline
//   double get_theta(){
//     return theta;
//   }

  
  inline
  double get_R_phi(){
    return R_phi;
  }

//   inline
//   double get_phi(){
//     return phi;
//   }
  
  inline
  double dotProduct(const double *s1, const double *s2){
    return s1[0] * s2[0] + s1[1] * s2[1] + s1[2] * s2[2];
  }

  
  void hitRealSurfaceInten(MTRand &MTrng,
			   const double *absorb_surface,
			   const double *rs_surface,
			   const double *rd_surface,
			   double &PathSurfaceLeft);
  
  void TravelInMediumInten(MTRand &MTrng,
			   VirtualSurface &obVirtual,
			   const double *kl_Vol,
			   const double *scatter_Vol,			   
			   const double *X,
			   const double *Y,
			   const double *Z,
			   const int *VolFeature,
			   double &PathLeft,
			   double &PathSurfaceLeft);
  
  

  ~ray();
  bool VIRTUAL; // when VIRTUAL == 1, hit on virtual surface
  bool dirChange;
  
private:
  int VolElementNo;
  int currentvIndex; // for volume index
  int futurevIndex;
  int hitSurfaceIndex;
  
  // VirtualSurface obVirtual;
  RealSurface *obReal;
  TopRealSurface obTop_ray;
  BottomRealSurface obBottom_ray;
  FrontRealSurface obFront_ray;
  BackRealSurface obBack_ray;
  LeftRealSurface obLeft_ray;
  RightRealSurface obRight_ray;
  
  void get_specular_s(double *spec_s);  
  int Ncx, Ncy, Ncz, ghostX, ghostY, ghostTB;
  int iIndex, jIndex, kIndex;
  int futureViIndex, futureVjIndex, futureVkIndex;
  
  // basic data for a ray
  double xemiss, yemiss, zemiss;
  double xhit, yhit, zhit;
  int hitSurfaceiIndex, hitSurfacejIndex, hitSurfacekIndex;
  double directionVector[3];
  double length;
  int surfaceFlag;

  double disMin, disX, disY, disZ;
  double xx[2], yy[2], zz[2], inv_directionVector[3];
  int sign[3];

  double R_phi, R_theta, R_xemiss, R_yemiss, R_zemiss;
   
};

#endif
