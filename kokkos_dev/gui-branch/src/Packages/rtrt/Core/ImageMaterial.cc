
#include "ImageMaterial.h"
#include "HitInfo.h"
#include "UVMapping.h"
#include "UV.h"
#include "Point.h"
#include "Ray.h"
#include "Object.h"
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

void ImageMaterial::read_image(char* filename)
{
    char buf[200];
    sprintf(buf, "%s.hdr", filename);
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
    ifstream indata(filename);
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
}
