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


#ifndef ray_H
#define ray_H

#include <vector>
#include <cstdlib>

#include "RNG.h"
#include "VirtualSurface.h"
#include "RealSurface.h"
#include "TopRealSurface.h"
#include "BottomRealSurface.h"
#include "FrontRealSurface.h"
#include "BackRealSurface.h"
#include "LeftRealSurface.h"
#include "RightRealSurface.h"

class RNG;
class VirtualSurface;
class RealSurface;
class TopRealSurface;
class BottomRealSurface;
class FrontRealSurface;
class BackRealSurface;
class LeftRealSurface;
class RightRealSurface;

using namespace std;

class ray {

public:

  ray(const int &_VolElementNo,
      const int &Ncx_, const int &Ncy_, const int &Ncz_,
      const int &offset_);

  
 //  void set_emissP_surf(const double &xlow, const double &xup,
// 		       const double &ylow, const double &yup,
// 		       const double &zlow, const double &zup);

  void set_emissP( const double &xlow, const double &xup,
		   const double &ylow, const double &yup,
		   const double &zlow, const double &zup);
		     
  
  void set_emissS_vol(double *sVol);
  
  // Travelling Ray's Emission Point
   // set hit point as next emission point,
  // and set hit point surface as new emission point surface
  void update_emissP();

  // checking which surfaces the ray get intersection in
  // Returns true if a surface was hit.
  bool surfaceIntersect( const double *X,
                         const double *Y,
                         const double *Z,
                         const int *VolFeature );
 
  void get_directionS(double *s);
  double ray_length();
  // int get_emissSurfaceIndex();
  int get_hitSurfaceIndex();
  int get_surfaceFlag() { return surfaceFlag; }
  
  void set_currentvIndex(const int &iIndex,
			 const int &jIndex,
			 const int &kIndex);
  
  void set_currentvIndex(const int &VolIndex);
  void update_vIndex();
  
  int get_currentvIndex();
  
  double get_xemiss();
  double get_yemiss();
  double get_zemiss();
  
  void hitRealSurfaceInten(const double *absorb_surface,
			   const double *rs_surface,
			   const double *rd_surface,
			   double &PathSurfaceLeft);
  
  void TravelInMediumInten(const double *kl_Vol,
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
  int offset;
  int currentvIndex; // for volume index
  int hitSurfaceIndex;
 
  VirtualSurface obVirtual;
  RealSurface *obReal;
  TopRealSurface obTop_ray;
  BottomRealSurface obBottom_ray;
  FrontRealSurface obFront_ray;
  BackRealSurface obBack_ray;
  LeftRealSurface obLeft_ray;
  RightRealSurface obRight_ray;
  
  RNG rng;  
  double dotProduct(const double *s1, const double *s2);
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
   
};

#endif
