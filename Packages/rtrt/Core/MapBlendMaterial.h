
#ifndef MAPBLENDMATERIAL_H
#define MAPBLENDMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>

namespace rtrt {

class MapBlendMaterial : public Material
{
  
 protected:

  Material *mat1_, *mat2_;
  PPMImage map_;

 public:

  MapBlendMaterial(const string& s, Material *one, Material *two)
    : mat1_(one), mat2_(two), map_(s) {}
  virtual ~MapBlendMaterial() {}

  virtual void shade(Color& result, const Ray &ray, const HitInfo &hit, 
                     int depth, double atten, const Color &accumcolor,
                     Context *cx)
  {
    UVMapping* map=hit.hit_obj->get_uvmapping();
    UV uv;
    Point hitpos(ray.origin()+ray.direction()*hit.min_t);
    map->uv(uv, hitpos, hit);
    Color final,original=result;
    double u=uv.u();
    double v=uv.v();
    double percent;
    unsigned width = map_.get_width();
    unsigned height = map_.get_height();

    if (!(map_.valid() && mat1_ && mat2_)) return;

    double tu = u-(unsigned)u;
    double tv = v-(unsigned)v;

    percent = (map_((unsigned)(tu*width),
                    (unsigned)(tv*height))).red();

    mat1_->shade(result,ray,hit,depth,atten,accumcolor,cx);
    final = result*percent;
    result = original;
    mat2_->shade(result,ray,hit,depth,atten,accumcolor,cx);
    final += result*(1-percent);

    result = final;
  }
};

} // end namespace

#endif

