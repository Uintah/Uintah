
#include <Packages/rtrt/Core/ImageMaterial.h>
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

Persistent* imageMaterial_maker() {
  return new ImageMaterial();
}

// initialize the static member type_id
PersistentTypeID ImageMaterial::type_id("ImageMaterial", "Material", 
					imageMaterial_maker);

ImageMaterial::ImageMaterial(int, const string &texfile, 
			     ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode, double Kd,
			     const Color& specular, double specpow,
			     double refl, bool /*flipped*/) :
  umode(umode), vmode(vmode), Kd(Kd), specular(specular),
  specpow(specpow), refl(refl),  transp(0), valid_(false)
{
    read_hdr_image(texfile);
    outcolor=Color(0,0,0);
}

ImageMaterial::ImageMaterial(const string &texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode, double Kd,
			     const Color& specular, double specpow,
			     double refl, bool flipped) :
  umode(umode), vmode(vmode), Kd(Kd), specular(specular),
  specpow(specpow), refl(refl),  transp(0), valid_(false)
{
  filename_ = texfile;  // Save filename, mostly for debugging.

  PPMImage ppm(texfile,flipped);
  if (ppm.valid())  {
    valid_=true;
    int nu, nv;
    ppm.get_dimensions_and_data(image, nu, nv);
  } else {
    cerr << "Error reading ImageMaterial: "<<texfile<<"\n";
    image.resize(3,3);
    image(0,0)=Color(1,0,1);
  }
  outcolor=Color(0,0,0);
}

ImageMaterial::ImageMaterial(const string &texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode, 
			     const Color& specular, double Kd, double specpow,
			     double refl,  double transp, 
			     bool flipped) :
  umode(umode), vmode(vmode), specular(specular), Kd(Kd),
  specpow(specpow), refl(refl),  transp(transp), valid_(false)
{
  PPMImage ppm(texfile,flipped);
  if (ppm.valid()) {
      valid_=true;
      int nu, nv;
      ppm.get_dimensions_and_data(image, nu, nv);
  } else {
    cerr << "Error reading ImageMaterial: "<<texfile<<"\n";
    image.resize(3,3);
    image(0,0)=Color(1,0,1);
  }
  outcolor=Color(0,0,0);
}

ImageMaterial::~ImageMaterial()
{
}

Color ImageMaterial::interp_color(Array2<Color>& image,
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

void ImageMaterial::shade(Color& result, const Ray& ray,
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
    switch(umode){
    case Nothing:
	if(u<0 || u>1){
	    diffuse=outcolor;
	    goto skip;
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
	    goto skip;
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

    diffuse = interp_color(image,u,v);

skip:
    phongshade(result, diffuse, specular, specpow, refl,
                ray, hit, depth,  atten,
               accumcolor, cx);
}

void
ImageMaterial::read_hdr_image(const string &filename)
{
   char buf[200];
   sprintf(buf, "%s.hdr", filename.c_str());
   ifstream in(buf);
   if(!in){
     cerr << "Error opening header: " << buf << '\n';
     exit(1);
   }
   int nu, nv;
   in >> nu >> nv;
   if(!in){
     cerr << "Error reading header: " << buf << '\n';
     exit(1);
   }
   ifstream indata(filename.c_str());
   image.resize(nu, nv);
   for(int i=0;i<nu;i++){
     for(int j=0;j<nv;j++){
       unsigned char color[3];
       indata.read((char*)color, 3);
       double r=color[0]/255.;
       double g=color[1]/255.;
       double b=color[2]/255.;
       image(i,j)=Color(r,g,b);
     }
   }
   if(!indata){
     cerr << "Error reading ImageMaterial: " << filename << "\n";
     image.resize(3,3);
     image(0,0)=Color(0.7,0.7,0.7);     
   }
  valid_ = true;
}

const int IMAGEMATERIAL_VERSION = 1;

void 
ImageMaterial::io(SCIRun::Piostream &str)
{
  str.begin_class("ImageMaterial", IMAGEMATERIAL_VERSION);
  Material::io(str);
  SCIRun::Pio(str, (unsigned int&)umode);
  SCIRun::Pio(str, (unsigned int&)vmode);
  SCIRun::Pio(str, Kd);
  SCIRun::Pio(str, specular);
  SCIRun::Pio(str, specpow);
  SCIRun::Pio(str, refl);
  SCIRun::Pio(str, transp);
  rtrt::Pio(str, image);
  SCIRun::Pio(str, outcolor);
  SCIRun::Pio(str, valid_);    
  SCIRun::Pio(str, filename_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::ImageMaterial*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::ImageMaterial::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::ImageMaterial*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
