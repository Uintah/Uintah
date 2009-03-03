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



#ifndef TILEIMAGEMATERIAL_H
#define TILEIMAGEMATERIAL_H 1

#include <Packages/rtrt/Core/ImageMaterial.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace rtrt {
class TileImageMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::TileImageMaterial*&);
}

namespace rtrt { 

using std::string;

class TileImageMaterial : public ImageMaterial {

public:
  TileImageMaterial(int /* oldstyle */, const string &filename, 
		    double Kd, const Color& specular,
		    int specpow, double refl=0, bool flipped=0);
  TileImageMaterial(const string &filename,
		    double Kd, const Color& specular,
		    int specpow, double refl, 
		    double transp=0, bool flipped=false);
  TileImageMaterial(const string &filename,
		    double Kd, const Color& specular,
		    int specpow, double refl=0, bool flipped=false);
  virtual ~TileImageMaterial() {}

  TileImageMaterial() : ImageMaterial() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, TileImageMaterial*&);

  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth, 
		     double atten, const Color& accumcolor,
		     Context* cx);
};

} // end namespace rtrt

#endif
