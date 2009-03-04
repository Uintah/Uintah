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


//========================================================================
//
// Filename: SHLambertianMaterial.cc
//
//
// Material used for rendering irradiance environment maps with 
// spherical harmonic coefficients.
//
//
//
//
// Reference: This is an implementation of the method described by
//            Ravi Ramamoorthi and Pat Hanrahan in their SIGGRAPH 2001 
//            paper, "An Efficient Representation for Irradiance
//            Environment Maps".
//
//========================================================================

#ifndef SHLAMBERTIANMATERIAL_H
#define SHLAMBERTIANMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {

class SHLambertianMaterial : public Material {

public:

  SHLambertianMaterial( const Color& R, char* envmap, float scale = 10.0, 
			int type = 1 );
  virtual ~SHLambertianMaterial( void );
  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  virtual void shade( Color& result, const Ray& ray,
		      const HitInfo& hit, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx );
    
private:

  Color irradCoeffs( const Vector& N ) const;

  Color albedo;
  Color L00, L1_1, L10, L11, L2_2, L2_1, L20, L21, L22;
  float fudgeFactor;

};

} // end namespace rtrt

#endif
