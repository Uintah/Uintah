
#ifndef MULTIMATERIAL_H
#define MULTIMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <vector>

using std::vector;

namespace rtrt {

struct MatPercent {
 public:
  Material *material;
  double percent;

  MatPercent(Material *m, double d) { material=m; percent=d; }
  ~MatPercent() {}
};

class MultiMaterial : public Material {

 protected:

  vector<MatPercent*> material_stack_;

 public:

  MultiMaterial() {}
  virtual ~MultiMaterial() {
    unsigned loop,length;
    length = material_stack_.size();
    for (loop=0; loop<length; ++loop) {
      delete material_stack_[loop];
    }
    material_stack_.resize(0);
  }

  unsigned push_back(Material *m, double percent)
  {
    material_stack_.push_back(new MatPercent(m,percent));
    return material_stack_.size();
  }

  virtual void shade(Color& result, const Ray &ray, const HitInfo &hit, 
                     int depth, double atten, const Color &accumcolor,
                     Context *cx)
  {
    // this can be really inefficient, should be improved
    unsigned loop,length;
    Color final;
    double percent;
    length = material_stack_.size();
    if (length>0) {
      material_stack_[0]->material->shade(result,ray,hit,depth,
                                         atten, accumcolor,cx);
      percent = material_stack_[0]->percent;
      final = result*percent;
      for (loop=1; loop<length; ++loop) {
        material_stack_[loop]->material->shade(result,ray,hit,depth,
                                               atten,accumcolor,cx);
        percent = (1-percent)*material_stack_[loop]->percent;
        final += result*percent;
      }
    }
  }
};

} // end namespace

#endif
