#ifndef ray_H
#define ray_H

#include <vector>
#include <cstdlib>

#include "RNG.h"
#include "VirtualSurface.h"
#include "RealSurface.h"

class RNG;
class VirtualSurface;

using namespace std;

class ray {

public:
  //bool Origin;
  ray(const int &_BottomStartNo,
      const int &_FrontStartNo,
      const int &_BackStartNo,
      const int &_LeftStartNo,
      const int &_RightStartNo,
      const int &_sumElementNo,
      const int &_totalElementNo,
      const int &_VolElementNo,
      const int &_LeftRightNo,
      const double &xc, const double &yc, const double &zc);

  // Original Emission Point
  void getEmissionPosition(const double &alow,
			   const double &aup,
			   const double &blow,
			   const double &bup,
			   const double &constv,
			   const int &surfaceIndex);


  void getEmissionPositionVol( const double &xlow, const double &xup,
			       const double &ylow, const double &yup,
			       const double &zlow, const double &zup,
			       const int &vIndex);
  
  void get_EmissSVol(double *sVol);
  
  // Travelling Ray's Emission Point
   // set hit point as next emission point,
  // and set hit point surface as new emission point surface
  void setEmissionPosition();

  // checking which surfaces the ray get intersection in
  double surfaceIntersect( const double *VolTable);
			  
  // functions to get these private data members out into public
  void get_emiss_point(double *emissP) const;
  void get_hit_point(double *hitP) const;
  void get_directionS(double *s);
  double ray_length();
  int get_emissSurfaceIndex();
  int get_hitSurfaceIndex();
  void set_currentvIndex(const int &vIndex);
  void set_currentIndex(const int &index);
  int get_currentIndex();
  int get_currentvIndex();
  
  


  double SurfaceEmissFlux(const int &offset_index,
			  const double *emiss_surface,
			  const double *T_surface,
			  const double *a_surface);
 

  
  double SurfaceIntensity(const int &offset_index,
			  const double *emiss_surface,
			  const double *T_surface,
			  const double *a_surface);
 


  double SurfaceEmissFluxBlack(const int &offset_index,
			       const double *T_surface,
			       const double *a_surface);
 

  
  double SurfaceIntensityBlack(const int &offset_index,
			       const double *T_surface,
			       const double *a_surface);

  
    
  double VolumeEmissFlux(const int &vIndex,
			 const double *kl_Vol,
			 const double *T_Vol,
			 const double *a_Vol);


  
  double VolumeIntensity(const int &vIndex,
			 const double *kl_Vol,
			 const double *T_Vol,
			 const double *a_Vol);
 



  double VolumeEmissFluxBlack(const int &vIndex,
			      const double *T_Vol,
			      const double *a_Vol);
 

  
  double VolumeIntensityBlack(const int &vIndex,
			      const double *T_Vol,
			      const double *a_Vol);
 


  void hitRealSurfaceInten(const double *absorb_surface,
			   const double *rs_surface,
			   const double *rd_surface,
			   vector<double> &PathSurfaceLeft);

  void TravelInMediumInten(const double *kl_Vol,
			   const double *scatter_Vol,			   
			   const double *VolNeighborArray,
			   const double *VolTableArray,
			   vector<double> &PathLeft,
			   vector<int> &PathIndex,
			   vector<double> &PathSurfaceLeft);

  
  void  get_integrInten(double *integrInten,
			const double *gg,
			const double *netInten,
			const int &ggNo, const int &local_ElementNo);
  
  ~ray();
  
private:
  static double pi;
  static double SB;
  int BottomStartNo, FrontStartNo, BackStartNo, LeftStartNo;
  int RightStartNo, sumElementNo, totalElementNo, VolElementNo, LeftRightNo;
  double X, Y, Z; // the dimension of the geometry
  double xemiss, yemiss, zemiss;
  double xhit, yhit, zhit;
  double directionVector[3];
  double length;
  double SurEmissFlux, SurInten;
  double VolEmissFlux, VolInten;
  int surfaceFlag;
  int currentvIndex; // for volume index
  int currentIndex; // for any current Index , surface or volume
  // indices of emission point surface and hit point surface
  int emissSurfaceIndex, hitSurfaceIndex;
  int emissVolIndex;
  //bool NoMedia;
  VirtualSurface obVirtual;
  RNG rng;  
  double dotProduct(double *s1, const double *s2);
  void get_specular_s(double *spec_s);
  
  // calculate T function
  // although xemiss, yemiss, zemiss is updating, we just use the right first set

    
};

#endif
