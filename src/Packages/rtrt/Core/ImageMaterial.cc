
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

ImageMaterial::ImageMaterial(int, const string &texfile, 
			     ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode, double Kd,
			     const Color& specular, double specpow,
			     double refl)
    : umode(umode), vmode(vmode), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(0), flip_(false), valid_(false)
{
    read_hdr_image(texfile);
    outcolor=Color(0,0,0);
}

ImageMaterial::ImageMaterial(const string &texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode, double Kd,
			     const Color& specular, double specpow,
			     double refl)
    : umode(umode), vmode(vmode), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(0), flip_(false), valid_(false)
{
  PPMImage ppm(texfile);
  int nu, nv;
  ppm.get_dimensions_and_data(image, nu, nv);
  outcolor=Color(0,0,0);
}

ImageMaterial::ImageMaterial(const string &texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode, double Kd,
			     const Color& specular, double specpow,
			     double refl,  double transp)
    : umode(umode), vmode(vmode), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(transp), flip_(false), 
      valid_(false)
{
  PPMImage ppm(texfile);
  int nu, nv;
  ppm.get_dimensions_and_data(image, nu, nv);
  outcolor=Color(0,0,0);
}

ImageMaterial::~ImageMaterial()
{
}

Color interp_color(Array2<Color>& image,
				  double u, double v)
{
    u *= (image.dim1()-1);
    v *= (image.dim2()-1);
    
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
    double u=uv.u();
    double v=uv.v();
    switch(umode){
    case None:
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
	if(u>=1)
	    u=1;
	else if(u<0)
	    u=0;
    };
    switch(vmode){
    case None:
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
	if(v>=1)
	    v=1;
	else if(v<0)
	    v=0;
    };
    {
      if (flip_)
	diffuse = interp_color(image,u,1-v);
      else
	diffuse = interp_color(image,u,v);
    }
skip:
    phongshade(result, diffuse, specular, specpow, refl,
                ray, hit, depth,  atten,
               accumcolor, cx);
}

void ImageMaterial::read_hdr_image(const string &filename)
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
     cerr << "Error reading image!\n";
     exit(1);
   }
  valid_ = true;
}
