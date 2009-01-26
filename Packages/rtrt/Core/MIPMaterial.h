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


#ifndef MIPMATERIAL_H
#define MIPMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/ImageMaterial.h>

#include <string>

using namespace std;
using rtrt::ImageMaterial;

namespace rtrt {
  class MIPMaterial;
}

/*  namespace SCIRun { */
/*    void Pio(Piostream&, rtrt::MIPMaterial*&); */
/*  } */

namespace rtrt {
class MIPMaterial : public Material
{
    
public:
    Color ambient;
    double Kd;
    Color specular;
    int specpow;
    double refl;
    string filename_;
    bool valid_;
    Color outcolor;
    Array2<Color> *image;
    int n_images;

    MIPMaterial() : Material() {} // for Pio.
    MIPMaterial(const string& filename,
                double Kd, const Color specular,
		int specpow, double refl=0,
		bool flipped=0);
    MIPMaterial(const string& filename, rtrt::ImageMaterial::Mode, 
                rtrt::ImageMaterial::Mode,
                double Kd, const Color specular,
		int specpow, double refl=0,
		bool flipped=0);

/*      static  SCIRun::PersistentTypeID type_id; */
/*      virtual void io(SCIRun::Piostream &stream); */
/*      friend void SCIRun::Pio(SCIRun::Piostream&, MIPMaterial*&); */

    ~MIPMaterial();
    virtual void shade(Color& result, const Ray& ray,
                       const HitInfo& hit, int depth, 
                       double atten, const Color& accumcolor,
                       Context* cx);
};
}
#endif
