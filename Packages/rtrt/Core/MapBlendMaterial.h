
#ifndef MAPBLENDMATERIAL_H
#define MAPBLENDMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>

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
    Color mapcolor,final,original=result;
    double u=uv.u();
    double v=uv.v();
    unsigned width = map_.get_width();
    unsigned height = map_.get_height();

    mapcolor = map_((unsigned)(u*width+.5),
                    (unsigned)(v*height+.5));

    double percent = mapcolor.red();

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
