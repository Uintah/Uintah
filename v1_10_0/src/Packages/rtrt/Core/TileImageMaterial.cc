
#include <Packages/rtrt/Core/TileImageMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>

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
				      const Color& specular, double specpow,
				      double refl, bool flipped/*=false*/ ) :
  ImageMaterial(1,texfile,ImageMaterial::Tile,ImageMaterial::Tile,
		Kd,specular,specpow,refl,flipped)
{
}

TileImageMaterial::TileImageMaterial( const string &texfile, 
				      double Kd,
				      const Color& specular, double specpow,
				      double refl, bool flipped /*= false*/ ) :
  ImageMaterial(texfile,ImageMaterial::Tile,ImageMaterial::Tile,
		specular,Kd,specpow,refl,flipped)
{
}

TileImageMaterial::TileImageMaterial(const string &texfile, 
                                     double Kd,
                                     const Color& specular, double specpow,
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
