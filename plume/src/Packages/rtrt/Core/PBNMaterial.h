
#ifndef PBNMATERIAL_H
#define PBNMATERIAL_H 1

// Paint-By-Numbers material

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <map>

using std::map;
using std::pair;

namespace rtrt {

class PBNMaterial : public Material
{

  typedef map<unsigned, Material*> mat_map;
  typedef mat_map::iterator        mat_iter;

protected:

  mat_map  material_map_;
  PPMImage pbnmap_;

public:

  PBNMaterial(const string& s) : pbnmap_(s) {}
  virtual ~PBNMaterial() {}

  virtual void io(SCIRun::Piostream &/*stream*/)
  { ASSERTFAIL("not implemented"); }
  void insert(Material *m, unsigned n)
  {
    mat_iter i = material_map_.find(n);
    if (i == material_map_.end())
      material_map_.insert(pair<unsigned,Material*>(n,m));
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

    mat_iter i = material_map_.find(index);
    if (i != material_map_.end())
      (*i).second->shade(result,ray,hit,depth,atten,accumcolor,cx);
  }
};

} // end namespace

#endif

  
