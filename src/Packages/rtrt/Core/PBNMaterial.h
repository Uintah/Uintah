
#ifndef PBNMATERIAL_H
#define PBNMATERIAL_H 1

// Paint-By-Numbers material

#include <Packages/rtrt/Core/Material.h>
#include <Core/Util/PPMImage.h>
#include <vector>

namespace rtrt {

class PBNMaterial
{

 protected:

  vector<Material*> material_stack_;
  PPMImage          pbnmap_;

 public:

  PBNMaterial(const string& s) : pbnmap_(s) {}
  virtual ~PBNMaterial() {}

  unsigned push_back(Material *m)
  {
    material_stack_.push_back(m);
    return material_stack_.size();
  }

  virtual void shade(Color& result, const Ray &ray, const HitInfo &hit, 
                     int depth, double atten, const Color &accumcolor,
                     Context *cx)
  {
    UVMapping* map=hit.hit_obj->get_uvmapping();
    UV uv;
    Point hitpos(ray.origin()+ray.direction()*hit.min_t);
    map->uv(uv, hitpos, hit);
    Color mapcolor;
    double u=uv.u();
    double v=uv.v();
    unsigned width = pbnmap_.get_width();
    unsigned height = pbnmap_.get_height();
    unsigned size = pbnmap_.get_size();

    mapcolor = pbnmap_(u*width,v*height);

    unsigned index = (unsigned)(mapcolor.red()*size+.5);
    if (index<material_stack_.size())
      material_stack_[index]->shade(result,ray,hit,depth,atten,accumcolor,cx);
  }
};

} // end namespace

#endif

  
