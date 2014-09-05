
#include <Packages/rtrt/Core/SharedTexture.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Stats.h>

#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Context.h>
#include <math.h>
#include <Packages/rtrt/Core/PPMImage.h>
#ifdef HAVE_PNG
#include <Packages/rtrt/Core/PNGImage.h>
#endif

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <stdlib.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

Persistent* sharedTexture_maker() {
  return new SharedTexture();
}

// initialize the static member type_id
PersistentTypeID SharedTexture::type_id("SharedTexture", "Material", 
					sharedTexture_maker);

SharedTexture::SharedTexture(const string &texfile, SharedTexture::Mode umode,
			     SharedTexture::Mode vmode, bool flipped) :
  umode(umode), vmode(vmode), valid_(false)
{
  filename_ = texfile;  // Save filename, mostly for debugging.
  unsigned long len = texfile.length();
  string extension = texfile.substr(len-3, 3);
  if(extension == "ppm") {
    PPMImage ppm(texfile,flipped);
    if (ppm.valid()) {
      valid_=true;
      int nu, nv;
      ppm.get_dimensions_and_data(image, nu, nv);
    }
  }
  else if(extension == "png") {
#ifdef HAVE_PNG
    PNGImage png(texfile,1);
    if (png.valid()) {
      valid_=true;
      int nu, nv;
      png.get_dimensions_and_data(image, alpha, nu, nv);
    }
#else
    cerr << "ERROR: Support for png images is not enabled.  Please configure with png support.\n";
    image.resize(3,3);
    image(0,0)=Color(1,0,1);
#endif
  }
  else {
    cerr << "Error reading SharedTexture: "<<texfile<<"\n";
    image.resize(3,3);
    image(0,0)=Color(1,0,1);
  }
  outcolor=Color(0,0,0);
}

SharedTexture::~SharedTexture()
{
}

Color SharedTexture::interp_color(Array2<Color>& image,
				  double u, double v)
{
  // u & v *= dimensions minus the slop(2) and the zero base difference (1)
  // for a total of 3
  u *= (image.dim1()-3);
  v *= (image.dim2()-3);
  
  int iu = (int)u;
  int iv = (int)v;
  
  double tu = u-iu;
  double tv = v-iv;
  
  Color c = image(iu,iv)*(1-tu)*(1-tv)+
    image(iu+1,iv)*tu*(1-tv)+
    image(iu,iv+1)*(1-tu)*tv+
    image(iu+1,iv+1)*tu*tv;
  
  return c;
}

float SharedTexture::return_alpha(Array2<float>& alpha, double u, double v)
{
  u *= (alpha.dim1()-3);
  v *= (alpha.dim2()-3);
  
  int iu = (int)u;
  int iv = (int)v;
  
  double tu = u-iu;
  double tv = v-iv;
  
  float alp = alpha(iu,iv)*(1-tu)*(1-tv)+
    alpha(iu+1,iv)*tu*(1-tv)+
    alpha(iu,iv+1)*(1-tu)*tv+
    alpha(iu+1,iv+1)*tu*tv;
  
  return alp;
}

void SharedTexture::shade(Color& result, const Ray& ray,
			  const HitInfo& hit, int depth, 
			  double atten, const Color& accumcolor,
			  Context* cx)
{
  UVMapping* map=hit.hit_obj->get_uvmapping();
  UV uv;
  Point hitpos(ray.origin()+ray.direction()*hit.min_t);
  map->uv(uv, hitpos, hit);
  Color diffuse = Color(0.0,0.0,0.0);
  Color  diffuse_temp;
  double u=uv.u()*uscale;
  double v=uv.v()*vscale;
  
  Ray rray(hitpos, ray.direction());
  HitInfo rhit;
  switch(umode){
  case Nothing:
    if(u<0 || u>1) {
      diffuse=outcolor;
      phongshade(result, diffuse, Color(0,0,0), 0, 0,
		 ray, hit, depth, atten,
		 accumcolor, cx);
      return;
    }
    break;
  case Tile:
    {
      int iu=(int)u;
      u-=iu;
      if (u < 0) u += 1;
    }
    break;
  case Clamp:
    if(u>1)
      u=1;
    else if(u<0)
      u=0;
  };
  switch(vmode){
  case Nothing:
    if(v<0 || v>1){
      diffuse=outcolor;
      phongshade(result, diffuse, Color(0,0,0), 0, 0,
		 ray, hit, depth, atten,
		 accumcolor, cx);
      return;
    }
    break;
  case Tile:
    {
      int iv=(int)v;
      v-=iv;
      if (v < 0) v += 1;
    }
    break;
  case Clamp:
    if(v>1)
      v=1;
    else if(v<0)
      v=0;
  };
  
  diffuse_temp = interp_color(image,u,v);

  if((alpha.dim1() == 0) && (alpha.dim2() == 0)) {
    // For ppm files
    diffuse = diffuse_temp;
  }
  else {
    // For png files
    float current_alpha = return_alpha(alpha, u, v);
    if(current_alpha == 1.0f) {
      diffuse = diffuse_temp;
    }
    else {
      // Send a ray
      diffuse = Color(0.0,0.0,0.0);
      double ratten = atten*(1.0-current_alpha); 
      Color rcolor;
      
      cx->worker->traceRay(rcolor, rray, depth+1, ratten,
			   accumcolor+diffuse, cx);
      diffuse+=rcolor;
      result=diffuse;
      return;
    }
  }

  phongshade(result, diffuse, Color(0,0,0), 0, 0,
	     ray, hit, depth,  atten,
	     accumcolor, cx);
}

const int SHAREDTEXTURE_VERSION = 1;

void 
SharedTexture::io(SCIRun::Piostream &str)
{
  str.begin_class("SharedTexture", SHAREDTEXTURE_VERSION);
  Material::io(str);
  SCIRun::Pio(str, (unsigned int&)umode);
  SCIRun::Pio(str, (unsigned int&)vmode);
  rtrt::Pio(str, image);
  SCIRun::Pio(str, outcolor);
  SCIRun::Pio(str, valid_);    
  SCIRun::Pio(str, filename_);
  str.end_class();
}

namespace SCIRun {
  void Pio(SCIRun::Piostream& stream, rtrt::SharedTexture*& obj)
  {
    SCIRun::Persistent* pobj=obj;
    stream.io(pobj, rtrt::SharedTexture::type_id);
    if(stream.reading()) {
      obj=dynamic_cast<rtrt::SharedTexture*>(pobj);
      //ASSERT(obj != 0)
    }
  }
} // end namespace SCIRun
