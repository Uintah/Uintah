
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <stdlib.h>

using namespace rtrt;
using namespace std;

ImageMaterial::ImageMaterial(char* texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode,
			     const Color& ambient, double Kd,
			     const Color& specular, double specpow,
			     double refl)
    : umode(umode), vmode(vmode), ambient(ambient), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(0)
{
    read_image(texfile);
    outcolor=Color(0,0,0);
}

ImageMaterial::ImageMaterial(char* texfile, ImageMaterial::Mode umode,
			     ImageMaterial::Mode vmode,
			     const Color& ambient, double Kd,
			     const Color& specular, double specpow,
			     double refl,  double transp)
    : umode(umode), vmode(vmode), ambient(ambient), Kd(Kd), specular(specular),
      specpow(specpow), refl(refl),  transp(transp)
{
    read_image(texfile);
    outcolor=Color(0,0,0);
}

ImageMaterial::~ImageMaterial()
{
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
	if(u>1)
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
	if(v>1)
	    v=1;
	else if(v<0)
	    v=0;
    };
    {
	u*=image.dim1();
	v*=image.dim2();
	int iu=(int)u;
	int iv=(int)v;
	diffuse=image(iu, iv);
    }
skip:
    phongshade(result, ambient, diffuse, specular, specpow, refl,
                ray, hit, depth,  atten,
               accumcolor, cx);
}

static void eat_comments_and_whitespace(ifstream &str)
{
  char c;
  str.get(c);
  while (1) {
    if (c==' '||c=='\t'||c=='\n') {
      str.get(c);
      continue;
    } else if (c=='#') {
      str.get(c);
      while(c!='\n')
        str.get(c);
    } else {
      str.unget();
      break;
    }
  }
}

void ImageMaterial::read_image(char* filename)
{
  unsigned nu, nv;
  double size;
  ifstream indata(filename);
  unsigned char color[3];
  string token;

  if (!indata.is_open()) {
    cerr << "ImageMaterial: WARNING: I/O fault: no such file: " << filename << endl;
  }
    

  indata >> token; // P6
  eat_comments_and_whitespace(indata);
  indata >> nu >> nv;
  eat_comments_and_whitespace(indata);
  indata >> size;
  eat_comments_and_whitespace(indata);
  image.resize(nu, nv);
  for(unsigned v=0;v<nv;++v){
    for(unsigned u=0;u<nu;++u){
      indata.read((char*)color, 3);
      double r=color[0]/size;
      double g=color[1]/size;
      double b=color[2]/size;
      image(u,v)=Color(r,g,b);
    }
  }
}


