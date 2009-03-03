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



#ifndef MULTIMATERIAL_H
#define MULTIMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace rtrt {
class MatPercent;
class MultiMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::MatPercent&);
void Pio(Piostream&, rtrt::MultiMaterial*&);
}

namespace rtrt {

struct MatPercent {
 public:
  Material *material;
  double percent;

  MatPercent(Material *m, double d) { material=m; percent=d; }
  ~MatPercent() {}

  friend void SCIRun::Pio(SCIRun::Piostream&, rtrt::MatPercent&);
};

class MultiMaterial : public Material {

 protected:

  std::vector<MatPercent*> material_stack_;

 public:

  MultiMaterial() {}
  virtual ~MultiMaterial() {
    size_t loop,length;
    length = material_stack_.size(); 
    for (loop=0; loop<length; ++loop) {
      delete material_stack_[loop];
    }
    material_stack_.resize(0);
  }

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, MultiMaterial*&);

  unsigned insert(Material *m, double percent)
  {
    material_stack_.push_back(new MatPercent(m,percent));
    return static_cast<unsigned int>(material_stack_.size());
  }

  void set(unsigned index, double percent) 
    { material_stack_[index]->percent = percent; }

  virtual void shade(Color& result, const Ray &ray, const HitInfo &hit, 
                     int depth, double atten, const Color &accumcolor,
                     Context *cx)
  {
    // this can be really inefficient, should be improved
    size_t loop,length;
    Color final,original=result;
    double percent;
    length = material_stack_.size();
    if (length>0) {
      material_stack_[0]->material->shade(result,ray,hit,depth,
                                         atten, accumcolor,cx);
      
      percent = material_stack_[0]->percent;
      if (percent<0) percent = 0;
      final = result*percent;
      for (loop=1; loop<length; ++loop) {
        result = original;
        material_stack_[loop]->material->shade(result,ray,hit,depth,
                                               atten,accumcolor,cx);
        percent = (1-percent)*material_stack_[loop]->percent;
        final += result*percent;
      }
    }
    result = final;
  }
};

} // end namespace

#endif
