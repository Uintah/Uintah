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


#ifndef RealSurface_H
#define RealSurface_H

#include <CCA/Components/Arches/MCRT/ArchesRMCRT/Surface.h>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/MersenneTwister.h>
#include <cmath>

class MTRand;
class ray;

class RealSurface : public Surface {
public:
  
  RealSurface();
  
  void get_s(MTRand &MTrng, double *s);
  
  virtual void set_n(double *nn) = 0;

  // given surfaceIndex, find limits of that surface element
  virtual void get_limits(const double *X,
			  const double *Y,
			  const double *Z) = 0;

  virtual void get_n() = 0;
  virtual void get_t1() = 0;
  virtual void get_t2() = 0;

  inline
  void set_theta(const double &random){
    theta = asin(sqrt(random));
  }

  inline
  double get_theta(){
    return theta;
  }

  
  inline
  double get_phi(){
    return phi;
  }

  inline
  double get_R_theta(){
    return R_theta;
  }


  inline
  double get_R_phi(){
    return R_phi;
  }

  
  inline
  int get_surfaceIndex(){
    return this->surfaceIndex;
  }


  inline
  double get_xlow(){
    return this->xlow;
  }


  inline
  double get_xup(){
    return this->xup;
  }

  inline
  double get_ylow(){
    return this->ylow;
  }


  inline
  double get_yup(){
    return this->yup;
  }
  
  inline
  double get_zlow(){
    return this->zlow;
  }


  inline
  double get_zup(){
    return this->zup;
  }
  

//   inline
//   int get_surfaceiIndex(){
//     return this->surfaceiIndex;
//   }
  

//   inline
//   int get_surfacejIndex(){
//     return this->surfacejIndex;
//   }

//   inline
//   int get_surfacekIndex(){
//     return this->surfacekIndex;
//   }
  
  
  double SurfaceEmissFlux(const int &i,
			  const double *emiss_surface,
			  const double *T_surface,
			  const double *a_surface);
			 
  double SurfaceEmissFluxBlack(const int &i,
			       const double *T_surface,
			       const double *a_surface);
  
  
  double SurfaceIntensity(const int &i,
			  const double *emiss_surface,
			  const double *T_surface,
			  const double *a_surface);

  
  double SurfaceIntensityBlack(const int &i,
			       const double *T_surface,
			       const double *a_surface);  
  
  friend class ray;
  
  ~RealSurface();

protected:

  double n[3], t1[3], t2[3];
  double xlow, xup, ylow, yup, zlow, zup;
  int surfaceIndex;
  int surfaceiIndex, surfacejIndex, surfacekIndex;
  double R_theta, R_phi;
 
};

#endif 
