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


#ifndef PHONGMATERIAL_H
#define PHONGMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>

namespace rtrt {
  class PhongMaterial;
}

namespace SCIRun {
  void Pio(Piostream&, rtrt::PhongMaterial*&);
}

namespace rtrt {

class PhongMaterial : public Material {
  Color Rd;   // diffuse reflectance
  double opacity;  // transparancy = 1 - opacity
  double Rphong;  // phong reflectance
  int phong_exponent;
public:
  PhongMaterial(const Color& Rd, double opacity, double Rphong = 0.0, 
		int phong_exponent = 100);
  inline Color get_diffuse() { return Rd; }
  inline void set_diffuse(const Color &d) { Rd = d; }
  inline double get_opacity() { return opacity; }
  inline void set_opacity(double o) { opacity = o; }
  inline double get_reflectance() { return Rphong; }
  inline void set_reflectance(double r) { Rphong = r; }
  inline int get_shininess() { return phong_exponent; }
  inline void set_shininess(int s) { phong_exponent = s; }
  virtual ~PhongMaterial();

  PhongMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, PhongMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
