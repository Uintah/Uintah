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



#include <Packages/rtrt/Core/TileImageMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/PPMImage.h>

#include <Core/Geometry/Point.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

#include <cstdio>
#include <cstdlib>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

Persistent* tileImageMaterial_maker() {
  return new TileImageMaterial();
}

// initialize the static member type_id
PersistentTypeID TileImageMaterial::type_id("TileImageMaterial", "Material", 
					    tileImageMaterial_maker);


TileImageMaterial::TileImageMaterial( int /*oldstyle*/, const string &texfile, 
				      double Kd,
				      const Color& specular, int specpow,
				      double refl, bool flipped/*=false*/ ) :
  ImageMaterial(1,texfile,ImageMaterial::Tile,ImageMaterial::Tile,
		Kd,specular,specpow,refl,flipped)
{
}

TileImageMaterial::TileImageMaterial( const string &texfile, 
				      double Kd,
				      const Color& specular, int specpow,
				      double refl, bool flipped /*= false*/ ) :
  ImageMaterial(texfile,ImageMaterial::Tile,ImageMaterial::Tile,
		specular,Kd,specpow,refl,flipped)
{
}

TileImageMaterial::TileImageMaterial(const string &texfile, 
                                     double Kd,
                                     const Color& specular, int specpow,
                                     double refl,  double transp, 
                                     bool flipped /*=false*/) :
  ImageMaterial(texfile,ImageMaterial::Tile,ImageMaterial::Tile,
		specular,Kd,specpow,refl,transp,flipped)
{
}

void TileImageMaterial::shade(Color& result, const Ray& ray,
                              const HitInfo& hit, int depth, 
                              double atten, const Color& accumcolor,
                              Context* cx)
{
  UVMapping* map=hit.hit_obj->get_uvmapping();
  UV uv;
  Point hitpos(ray.origin()+ray.direction()*hit.min_t);
  map->uv(uv, hitpos, hit);
  Color diffuse;
  double u=uv.u()*uscale;
  double v=uv.v()*vscale;
  
  int iu=(int)u;
  u-=iu;
  if (u < 0) u += 1;
  int iv=(int)v;
  v-=iv;
  if (v < 0) v += 1;
  
  diffuse = interp_color(image,u,v);
  
  phongshade(result, diffuse, specular, specpow, refl,
             ray, hit, depth,  atten,
             accumcolor, cx);
}

const int TILEIMAGEMATERIAL_VERSION = 1;

void 
TileImageMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("TileImageMaterial", TILEIMAGEMATERIAL_VERSION);
  ImageMaterial::io(str);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::TileImageMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::TileImageMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::TileImageMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
