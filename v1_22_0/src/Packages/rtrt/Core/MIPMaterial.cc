#include <Packages/rtrt/Core/MIPMaterial.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Core/Math/MinMax.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/UVMapping.h>
#include <Packages/rtrt/Core/UV.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Packages/rtrt/Core/Object.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

MIPMaterial::MIPMaterial(const string &texfile, 
			 double Kd, const Color specular,
			 double specpow, double refl,
			 bool flipped)
  : Kd(Kd), specular(specular),
    specpow(specpow), refl(refl)
{
  filename_ = texfile;
  
  // read the image
  PPMImage ppm(texfile,flipped);
  if (ppm.valid()) {
    valid_ = true;
    int nu,nv;
    // stick the image data into an array.
    
    Array2<Color>full_img;
    
    ppm.get_dimensions_and_data(full_img,nu,nv);
    n_images = (int)floor(log((double)Min(nu,nv))/log((double)2.)) + 1;
    
    image = new Array2<Color>[n_images];
    
    image[0].resize(nu,nv);
    
    for (int i=0; i<nu; i++)
      for (int j=0; j<nv; j++)
        image[0](i,j) = full_img(i,j);
    
    for (int i=1; i<n_images; i++)
      {
        image[i].resize(nu >> i, nv >> i);
      }
    
    // build the multires images
    for (int k=1; k<n_images; k++)
      for (int i=0; i<image[k].dim1(); i++)
        for (int j=0; j<image[k].dim2(); j++)
          {
            // BOX FILTERING!!!!!!
            image[k](i,j) = .25 *
              (image[k-1](2*i,  2*j) +
               image[k-1](2*i+1,2*j) +
               image[k-1](2*i,  2*j+1) +
               image[k-1](2*i+1,2*j+1));
          } 
  } else {
    cerr << "Error reading MIPMaterial: "<<texfile<<"\n";
    image[0].resize(3,3);
    image[0](0,0) = Color(1,0,1);
  }
  outcolor = Color(0,0,0);
  
}
MIPMaterial::MIPMaterial(const string &texfile, ImageMaterial::Mode , 
                         ImageMaterial::Mode,
			 double Kd, const Color specular,
			 double specpow, double refl,
			 bool flipped)
  : Kd(Kd), specular(specular),
    specpow(specpow), refl(refl)
{
  filename_ = texfile;
  
  // read the image
  PPMImage ppm(texfile,flipped);
  if (ppm.valid()) {
    valid_ = true;
    int nu,nv;
    // stick the image data into an array.

    Array2<Color>full_img;

    ppm.get_dimensions_and_data(full_img,nu,nv);
    n_images = (int)floor(log((double)Min(nu,nv))/log((double)2.)) + 1;

    image = new Array2<Color>[n_images];
    
    image[0].resize(nu,nv);

    for (int i=0; i<nu; i++)
      for (int j=0; j<nv; j++)
        image[0](i,j) = full_img(i,j);

    for (int i=1; i<n_images; i++)
      {
        image[i].resize(nu >> i, nv >> i);
      }

    // build the multires images
    for (int k=1; k<n_images; k++)
	for (int i=0; i<image[k].dim1(); i++)
	    for (int j=0; j<image[k].dim2(); j++)
	    {
              // BOX FILTERING!!!!!!
              image[k](i,j) = .25 *
                (image[k-1](2*i,  2*j) +
                 image[k-1](2*i+1,2*j) +
                 image[k-1](2*i,  2*j+1) +
                 image[k-1](2*i+1,2*j+1));
	    } 
  } else {
    cerr << "Error reading MIPMaterial: "<<texfile<<"\n";
    image[0].resize(3,3);
    image[0](0,0) = Color(1,0,1);
  }
  outcolor = Color(0,0,0);

}

MIPMaterial::~MIPMaterial()
{}


void MIPMaterial::shade(Color& result, const Ray& ray,
			const HitInfo& hit, int depth, 
			double atten, const Color& accumcolor,
			Context* cx)
{
  if (!valid_) {
    result = Color(1,0,1);
    return;
  }

    UVMapping* map=hit.hit_obj->get_uvmapping();
    UV uv;

    // calculate distance of hitpoint
    Point origin = ray.origin();
    Point hitpt = origin + ray.direction()*hit.min_t;
    Vector normal = hit.hit_obj->normal(hitpt,hit);

    int which = cx->worker->rendering_scene;
    
    // may not get the correct camera for one iteration!!!
    Camera* cam = cx->scene->get_camera(which);
    double fov = cam->get_fov();

    Point eye = cam->get_eye();

    // Image plane is MxM
    Image* img_plane = cx->scene->get_image(which);
    int M = img_plane->get_xres();

    // angle subtended by a pixel
    double theta = (double)fov/M;

    int N = Min(image[0].dim1(),image[0].dim2());
    
    BBox bbox;
    hit.hit_obj->compute_bounds(bbox,0);

    // total Height of textured plane
    double H = bbox.diagonal().length();
    
    map->uv(uv, hitpt, hit);
    Color diffuse;

    // distance to hit point
    double d = (hitpt-eye).length();
    
    // texels per pixel
    double n = 4 * theta;
    // height of textured plane seen thru pixel
    double h = 2. * d * tan(.5*theta*M_PI/180.);

    // number of levels in MIP pyramid
    int L = n_images-1;
    
    // calculate texture to use
    double l = log((double)h*N/(H*n)) / log((double)2.);

//  	fprintf(stderr, "Here we are: %lf\n",h);
    
    if (l > 0. && l < L)
    {
	int il = (int) l;
	double t = l - il;

	int iu0 = (int) (uv.u() * image[il].dim1());
	int iv0 = (int) (uv.v() * image[il].dim2());

	int iu1 = (int) (uv.u() * image[il+1].dim1());
	int iv1 = (int) (uv.v() * image[il+1].dim2());

	diffuse = (1.-t)*image[il](iu0,iv0) + t*image[il+1](iu1,iv1);
//  	double blend = (double)il/L;
	
//  	diffuse =
//  	    blend * Color(1,0,0) +
//  	    (1.-blend) * Color(0,0,1);
	
	
    } else if (l <= 0)
    {
	int iu0 = (int) (uv.u() * image[0].dim1());
	int iv0 = (int) (uv.v() * image[0].dim2());

	diffuse = image[0](iu0,iv0);
//  	diffuse = Color(0,0,1);
	
	
    } else 
    {
	int iu0 = (int) (uv.u() * image[L].dim1());
	int iv0 = (int) (uv.v() * image[L].dim2());

	diffuse = image[L](iu0,iv0);
//  	diffuse = Color(1,0,0);
	
    }
    
//      double theta = acos(normal.dot(ray.direction()));
//      double angle_param = fabs(theta-M_PI_2)/M_PI_2;

//      angle_param = pow(angle_param,.3);

    // POINT SAMPLING IN T!!!!
//      int l = (int)((image.size()-1)*(d/(maxd*angle_param)));
//      if (l > image.size()-1)
//  	l = image.size()-1;
    
//      int iu = (int) (uv.u() * image[l].dim1());
//      int iv = (int) (uv.v() * image[l].dim2());

//      diffuse = image[l](iu,iv);


    // LINEAR INTERPOLATION IN T!!!!
//      double l = ((image.size()-1)*(d/(maxd*angle_param)));
//      if (l > image.size()-1)
//  	l = image.size()-1;

//      int il = (int) l;

//      if (il < image.size()-1)
//      {
//  	double t = l - il;

//  	int iu0 = (int) (uv.u() * image[il].dim1());
//  	int iv0 = (int) (uv.v() * image[il].dim2());

//  	int iu1 = (int) (uv.u() * image[il+1].dim1());
//  	int iv1 = (int) (uv.v() * image[il+1].dim2());

//  	diffuse = (1.-t)*image[il](iu0,iv0) + t*image[il+1](iu1,iv1);
//      }
//      else 
//      {
//  	int iu = (int) (uv.u() * image[image.size()-1].dim1());
//  	int iv = (int) (uv.v() * image[image.size()-1].dim2());

//  	diffuse = image[image.size()-1](iu,iv);
//      }
    
    phongshade(result,  diffuse, specular, specpow, refl,
	       ray, hit, depth,  atten,
               accumcolor, cx);
}
