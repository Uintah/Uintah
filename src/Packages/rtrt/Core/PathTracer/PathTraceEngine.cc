

#include <Packages/rtrt/Core/PathTracer/PathTraceEngine.h>

#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Point2D.h>
#include <Packages/rtrt/Core/HitInfo.h>

#include <Core/Thread/Runnable.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <math.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

PathTraceContext::PathTraceContext(const Color &color,
				   const PathTraceLight &light,
				   Object* geometry,
                                   int num_samples, int max_depth):
  color(color), light(light), geometry(geometry), num_samples(num_samples),
  max_depth(max_depth)
{
  // Fix num_samples to be a complete square
  num_samples_root = (int)(ceil(sqrt((double)num_samples)));
  int new_num_samples = num_samples_root*num_samples_root;
  if (new_num_samples != this->num_samples) {
    cerr << "Changing the number of samples from "<<num_samples<<" to "<<new_num_samples<<"\n";
    this->num_samples = new_num_samples;
  }

  // Determine the scratch size needed
  float bvscale=0.3;
  pp_size=0;
  pp_scratchsize=0;
  geometry->preprocess(bvscale, pp_size, pp_scratchsize);
}

PathTraceWorker::PathTraceWorker(Group *group, PathTraceContext *ptc, char *texname):
  ptc(ptc), local_spheres(group), rng(10)
{
  // Generate groups of random samples
  double inc = 1./ptc->num_samples_root;
  for(int sgindex=0;sgindex<NUM_SAMPLE_GROUPS;sgindex++) {
    // Resize our sample bucket
    sample_points[sgindex].resize(ptc->num_samples);

    // This is our sample index
    int index=0;
    // u and v are offsets into the texel
    double u=0;
    for (int i=0;i<ptc->num_samples_root;i++) {
      ASSERT(index<ptc->num_samples);
      double v=0;
      for (int j=0;j<ptc->num_samples_root;j++) {
	sample_points[sgindex][index] = Point2D(u + inc*rng(),
						v + inc*rng());
	
	index++;
        v+=inc;
      }
      u+=inc;
    }
  }

  // Allocate the DepthStats and PerProcessor context
  ppc = new PerProcessorContext(ptc->pp_size, ptc->pp_scratchsize);
  depth_stats = new DepthStats();

  // Copy base filename
  basename=strdup(texname);
}

PathTraceWorker::~PathTraceWorker() {
  if (ppc) delete ppc;
  if (depth_stats) delete depth_stats;
}

void PathTraceWorker::run() {
  float inv_num_samples=1./ptc->num_samples;
  
  // Iterate over our spheres
  for(int sindex = 0; sindex < local_spheres->objs.size(); sindex++) {
    TextureSphere *sphere =
      dynamic_cast<TextureSphere*>(local_spheres->objs[sindex]);
    if (!sphere) {
      // Something has gone horribly wrong
      cerr << "Object in local list (index="<<sindex<<") is not a TextureSphere\n";
      continue;
    }
    
    // Iterate over each texel
    int width=sphere->texture.dim1();
    int height=sphere->texture.dim2();
    double inv_width=1./width;
    double inv_height=1./height;
    for(int v=0;v<height;v++)
      for(int u=0;u<width;u++) {
	int sgindex = (int)(rng()*(NUM_SAMPLE_GROUPS-1));
	int sgindex2 = (int)(rng()*(NUM_SAMPLE_GROUPS-1));
	
        for(int sample=0;sample<ptc->num_samples;sample++) {
          Point2D sample_point = sample_points[sgindex][sample];
	  
          // Project sample point onto sphere
	  Point origin;
	  Vector normal;
	  {
	    double phi=M_PI*inv_width*((u+sample_point.x()));
	    double theta=M_PI*(1-2*inv_height*(v+sample_point.y()));
	    double x=cos(phi)*sin(theta);
	    double y=sin(phi)*sin(theta);
	    double z=cos(theta);
	    
	    origin=sphere->cen + sphere->radius*Vector(x,y,z);
	    normal=Vector(x,y,z);
	  }

	  Color result(0,0,0);
	  PathTraceLight* light=&(ptc->light);
	  for (int depth=0;depth<=ptc->max_depth;depth++) {
	    // Compute direct illumination (assumes one light source)
            Point random_point=light->random_point(rng(),rng(),rng());

	    // Shadow ray
	    Vector sr_dir=random_point-origin;
	    double distance=sr_dir.normalize();
	    double cosine=SCIRun::Dot(sr_dir, normal);
            if(cosine>0.0) {
	      Ray s_ray(origin, sr_dir);
	      HitInfo s_hit;
	      s_hit.min_t=distance;
	      Color s_color(1,1,1);
	      
	      ptc->geometry->light_intersect(s_ray, s_hit, s_color, depth_stats, ppc);
	      if (!s_hit.was_hit)
		result+=ptc->color*light->color*light->area*(cosine/(distance*distance*M_PI));
	    }
			      
	    if (depth==ptc->max_depth)
	      break;
	    
	    // Pick a random direction on the hemisphere
	    // We should really sample the entire hemisphere of directions
	    Vector v0(Cross(normal, Vector(1,0,0)));
	    if(v0.length2()==0)
	      v0=Cross(normal, Vector(0,1,0));
	    Vector v1=Cross(normal, v0);
	    Vector v2=Cross(normal, v1);
	    
	    Vector v_out;
	    {
	      Point2D hemi_sample=sample_points[sgindex2][sample];
	      double phi=2.0*M_PI*hemi_sample.x();
	      double r=sqrt(hemi_sample.y());
	      double x=r*cos(phi);
	      double y=r*sin(phi);
	      double z=sqrt(1.0-x*x-y*y);
	      v_out=Vector(x,y,z);
	    }
	    Vector dir=v0*v_out.x()+v1*v_out.y()+v2*v_out.z();
	    
	    Ray ray(origin, dir);
#if 0          
	    // Trace ray into our local spheres
	    HitInfo local_hit;
	    local_spheres->intersect(ray, local_hit, depth_stats, ppc);
	    
	    // Trace ray into geometry
	    HitInfo global_hit;
	    ptc->geometry->intersect(ray, global_hit, depth_stats, ppc);
	    
	    // Resolve next sphere
	    //  if local, set flag to indicate that we can accumulate
	    //    the next result in its texture (?)
	    //  etc., etc.
#else
	    // Trace ray into geometry
	    HitInfo global_hit;
	    ptc->geometry->intersect(ray, global_hit, depth_stats, ppc);
	    
	    if (global_hit.was_hit) {
	      // Set next origin/normal
	      origin=origin+global_hit.min_t*dir;
	      normal=global_hit.hit_obj->normal(origin, global_hit);
	    }
	    else {
	      // Accumulate bg color?
	      break;
	    }
#endif
	  } // end depth
	  
          // Store result
	  sphere->texture(u,v)+=result;
	} // end sample
	
        // Normalize result
	sphere->texture(u,v)=inv_num_samples*sphere->texture(u,v);
      } // end texel
  } // end sphere
  
  // Write out textures
  for(int sindex = 0; sindex < local_spheres->objs.size(); sindex++) {
    TextureSphere *sphere  =
      dynamic_cast<TextureSphere*>(local_spheres->objs[sindex]);
    if (!sphere) {
      // Something has gone horribly wrong
      continue;
    }

    sphere->writeTexture(basename,sindex);
  }
}

TextureSphere::TextureSphere(const Point &cen, double radius, int tex_res):
  Sphere(0, cen, radius), texture(tex_res, tex_res)
{
  texture.initialize(0);
}

void TextureSphere::writeTexture(char* basename, int index)
{
  // Create the filename
  char *buf = new char[strlen(basename) + 20];
  sprintf(buf, "%s%05d.nrrd", basename, index);
  FILE *out = fopen(buf, "wb");
  if (!out) {
    cerr << "Cannot open "<<buf<<" for writing\n";
    return;
  }
  
  int width = texture.dim1();
  int height = texture.dim2();
  fprintf(out, "NRRD0001\n");
  fprintf(out, "type: float\n");
  fprintf(out, "dimension: 3\n");
  fprintf(out, "sizes: 3 %d %d\n", width, height);
  fprintf(out, "spacings: NaN 1 1\n");
  fprintf(out, "endian: big\n");
  fprintf(out, "encoding: raw\n");
  fprintf(out, "\n");
  // Iterate over each texel
  for(int v = 0; v < height; v++)
    for(int u = 0; u < width; u++) {
      float data[3];
      data[0] = texture(u,v).red();
      data[1] = texture(u,v).green();
      data[2] = texture(u,v).blue();
      if (fwrite(data, sizeof(float), 3, out) != 3) {
	cerr << "Trouble writing texel for sphere "<<index<<" at ["<<u<<", "<<v<<"]\n";
	return;
      }
    }
  fclose(out);
}

PathTraceLight::PathTraceLight(const Point &cen, double radius,
			       const Color &c):
  center(cen), radius(radius), color(c)
{
  area = 2.*M_PI*radius;
}

Point PathTraceLight::random_point(double r1, double r2, double r3)
{
  return center;

  double theta = 2.*M_PI*r1;
  double alpha = 2.*M_PI*r2;
  double weight1 = sqrt(r3);
  double weight2 = sqrt(1.0 - r3);
  return Vector(weight2*sin(theta),
		weight2*cos(theta),
		weight1*sin(alpha)).normal().asPoint();
}
