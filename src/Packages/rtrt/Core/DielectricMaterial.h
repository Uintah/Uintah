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



#ifndef DIELECTRIC_H
#define DIELECTRIC_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
  class DielectricMaterial;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::DielectricMaterial*&);
}

namespace rtrt {

class DielectricMaterial : public Material {
  double n_in;          // refractive index of media normal points away from
  double n_out;         // refractive index of media normal points to
  double R0;            // reflectance at normal incidence
  int phong_exponent;
  Color extinction_in;  // transmittance through one unit distance of material:
  Color extinction_out;
  bool nothing_inside;	/* True if this object is empty - 
			 * optimize the recursive hits...*/
  double extinction_scale;    // Allow for a scale of t

  Color bg_out;         // exctinction_in to the infinite power
  Color bg_in;          // exctinction_out to the infinite power

public:
  DielectricMaterial(double n_in, double n_out, bool nothing_inside=false);
  DielectricMaterial(double n_in, double n_out, double R0, 
		     int phong_exponent, const Color& extinction_in,  
		     const Color& extinction_out,
		     bool nothing_inside=false, double extinction_scale=1);
  virtual ~DielectricMaterial();

  DielectricMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, DielectricMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
