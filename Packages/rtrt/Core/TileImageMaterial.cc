
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
		Kd,specular,specpow,refl,flipped)
{
}

TileImageMaterial::TileImageMaterial(const string &texfile, 
                                     double Kd,
                                     const Color& specular, double specpow,
                                     double refl,  double transp, 
                                     bool flipped /*=false*/) :
  ImageMaterial(texfile,ImageMaterial::Tile,ImageMaterial::Tile,
		Kd,specular,specpow,refl,transp,flipped)
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

