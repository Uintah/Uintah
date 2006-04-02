
#ifndef MAPBLENDMATERIAL_H
#define MAPBLENDMATERIAL_H 1

#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>

namespace rtrt {
class MapBlendMaterial;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::MapBlendMaterial*&);
}

namespace rtrt {

class MapBlendMaterial : public Material
{
 protected:

  Material *mat1_, *mat2_;
  PPMImage map_;

 public:

  MapBlendMaterial(const string& s, Material *one, Material *two, 
                   bool flip=false)
    : mat1_(one), mat2_(two), map_(s,flip) {}
  virtual ~MapBlendMaterial() {}

  MapBlendMaterial() : Material() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, MapBlendMaterial*&);

  Color interp_color(double u, double v)
  { 
    u *= (map_.get_width()-1);
    v *= (map_.get_height()-1);
    
    int iu = (int)u;
    int iv = (int)v;

    double tu = u-iu;
    double tv = v-iv;

    Color c = map_(iu,iv)*(1-tu)*(1-tv)+
	map_(iu+1,iv)*tu*(1-tv)+
	map_(iu,iv+1)*(1-tu)*tv+
	map_(iu+1,iv+1)*tu*tv;

    return c;
  }

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

    if (!(map_.valid() && mat1_ && mat2_)) return;

    double tu = u-(unsigned)u;
    double tv = v-(unsigned)v;

    percent = (interp_color(tu,tv)).red();

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

