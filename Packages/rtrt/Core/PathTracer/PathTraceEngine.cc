

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
  light(light), color(color), geometry(geometry), num_samples(num_samples),
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
	  double reflected_surface_dot = 1;
	  {
	    // range of (u+sx)/w [0,1]
	    // range of 2*M_PI * (u+sx)/w [0, 2*M_PI]
	    double phi=2*M_PI*inv_width*((u+sample_point.x()));
	    // range of (v+sy)/h [0,1]
	    // range of M_PI * (v+sy)/h [0, M_PI]
	    double theta=M_PI*inv_height*((v+sample_point.y()));
	    double x=cos(phi)*sin(theta);
	    double z=sin(phi)*sin(theta);
	    double y=cos(theta);
	    
	    origin=sphere->cen + sphere->radius*Vector(x,y,z);
	    normal=Vector(x,y,z).normal();
	  }

	  Color result(0,0,0);
	  PathTraceLight* light=&(ptc->light);
	  for (int depth=0;depth<=ptc->max_depth;depth++) {
	    // Compute direct illumination (assumes one light source)
            Vector random_dir=light->random_vector(rng(),rng(),rng());
	    Point random_point = light->point_from_normal(random_dir);
	    
	    // Shadow ray
	    Vector sr_dir=random_point-origin;

	    // Let's make sure the point on our light source is facing us

	    // Compute the dot product of the shadow ray and light normal
	    double light_norm_dot_sr = SCIRun::Dot(random_dir, -sr_dir);
	    if (light_norm_dot_sr < 0) {
	      // flip our random direction
	      random_dir = -random_dir;
	      // Update all the stuff that depended on it
	      random_point = light->point_from_normal(random_dir);
	      sr_dir = random_point - origin;
	      light_norm_dot_sr = SCIRun::Dot(random_dir, -sr_dir);
	    }
	    
	    double distance=sr_dir.normalize();
	    double normal_dot_sr=SCIRun::Dot(sr_dir, normal);
            if(normal_dot_sr>0.0) {
	      Ray s_ray(origin, sr_dir);
	      HitInfo s_hit;
	      s_hit.min_t=distance;
	      Color s_color(1,1,1);
	      
	      ptc->geometry->light_intersect(s_ray, s_hit, s_color,
					     depth_stats, ppc);
	      if (!s_hit.was_hit)
		result +=
		  ptc->color *
		  reflected_surface_dot *
		  light->color *
		  light_norm_dot_sr *
		  light->area *
		  (normal_dot_sr/(distance*distance*M_PI));
	    }
			      
	    if (depth==ptc->max_depth)
	      break;
	    
	    // Pick a random direction on the hemisphere
	    Vector v0(Cross(normal, Vector(1,0,0)));
	    if(v0.length2()==0)
	      v0=Cross(normal, Vector(0,1,0));
	    v0.normalize();
	    Vector v1=Cross(normal, v0);
	    v1.normalize();
	    
	    Vector v_out;
	    {
	      Point2D hemi_sample=sample_points[sgindex2][sample];
	      double phi=2.0*M_PI*hemi_sample.x();
	      double r=sqrt(hemi_sample.y());
	      double x=r*cos(phi);
	      double y=r*sin(phi);
	      double z=sqrt(1.0-x*x-y*y);
	      v_out=Vector(x,y,z).normal();
	    }
	    Vector dir=v0*v_out.x()+v1*v_out.y()+normal*v_out.z();
	    dir.normalize();
	    
	    Ray ray(origin, dir);

	    // Trace ray into geometry
	    HitInfo global_hit;
	    ptc->geometry->intersect(ray, global_hit, depth_stats, ppc);

	    if (global_hit.was_hit) {
	      // Set next origin/normal/reflected_surface_dot
	      origin = ray.origin()+global_hit.min_t*ray.direction();
	      normal = global_hit.hit_obj->normal(origin, global_hit);
	      reflected_surface_dot = SCIRun::Dot(normal, -ray.direction());
 	      if (reflected_surface_dot < 0) {
		// This could be due to two reasons.
		//
		// 1.  The surface point is inside another sphere and
		// we hit the inside of it.  If were inside another
		// sphere we aren't going to get any more light.
		//
		// 2.  Some really weird case where the point on our
		// sphere happens to be coincident with a neighboring
		// sphere and you somehow do some crazy math.  We
		// don't like negative dot products, so we kill it.
		break;
	      }
	    }
	    else {
	      // Accumulate bg color?
	      break;
	    }
	  } // end depth
	  
          // Store result
	  sphere->texture(u,v)+=result;
	} // end sample
	
        // Normalize result
	sphere->texture(u,v)=inv_num_samples*sphere->texture(u,v);
      } // end texel
    cout << "Finished sphere "<<sindex<<"\n";
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
  texture.initialize(Color(0,0,0));
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
#ifdef __sgi
  fprintf(out, "endian: big\n");
#else
  fprintf(out, "endian: little\n");
#endif
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

Vector PathTraceLight::random_vector(double r1, double r2, double r3)
{
  double theta = 2.*M_PI*r1;
  double alpha = 2.*M_PI*r2;
  double weight1 = sqrt(r3);
  double weight2 = sqrt(1.0 - r3);
  return Vector(weight2*sin(theta),
		weight2*cos(theta),
		weight1*sin(alpha)).normal();
}

Point PathTraceLight::random_point(double r1, double r2, double r3) {
  return point_from_normal(random_vector(r1, r2, r3));
}

Point PathTraceLight::point_from_normal(const Vector &dir) {
  return (center + dir*radius);
}

Vector PathTraceLight::normal(const Point &point) {
  return (point-center)/radius;
}
