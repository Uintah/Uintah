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

  
