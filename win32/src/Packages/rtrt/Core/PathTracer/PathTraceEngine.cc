

#include <Packages/rtrt/Core/PathTracer/PathTraceEngine.h>

#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Point2D.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Background.h>

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <math.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

PathTraceContext::PathTraceContext(float luminance,
				   const PathTraceLight &light,
				   Object* geometry,
				   Background *background,
                                   int num_samples_in, int num_sample_divs_in,
				   int max_depth, bool dilate,
				   int support, int use_weighted_ave,
				   float threshold, Semaphore *sem) :
  light(light), luminance(luminance), compute_shadows(true),
  compute_directlighting(true), geometry(geometry),
  background(background), num_sample_divs(num_sample_divs_in),
  max_depth(max_depth), dilate(dilate), support(support),
  use_weighted_ave(use_weighted_ave), threshold(threshold), sem(sem)
{
  if (num_sample_divs <= 0) {
    cerr << "num_sample_divs is less than or equal to zero.  "
	 << "Setting to one.\n";
    num_sample_divs = 1;
  }
  num_samples_root.resize(num_sample_divs);
  num_samples.resize(num_sample_divs);
  
  // Fix num_samples to be a complete square
  num_samples[0] = num_samples_in;
  num_samples_root[0] = (int)(ceil(sqrt((double)num_samples[0])));
  int new_num_samples = num_samples_root[0]*num_samples_root[0];
  if (new_num_samples != num_samples[0]) {
    cerr << "Changing the number of samples from "<<num_samples[0]<<" to "<<new_num_samples<<"\n";
    num_samples[0] = new_num_samples;
  }
  for(int i = 1; i < num_sample_divs; i++) {
    num_samples_root[i] = (int)ceil(num_samples_root[0] * (1 - (float)i/num_sample_divs));
    if (num_samples_root[i] < 5) num_samples_root[i] = 5;
    cout << "num_samples_root[i="<<i<<"] = "<<num_samples_root[i]<<"\n";
    num_samples[i] = num_samples_root[i] * num_samples_root[i];
  }

  // Determine the scratch size needed
  float bvscale=0.3;
  pp_size=0;
  pp_scratchsize=0;
  geometry->preprocess(bvscale, pp_size, pp_scratchsize);
}

PathTraceWorker::PathTraceWorker(Group *group, PathTraceContext *ptc,
				 char *texname, size_t offset):
  ptc(ptc), local_spheres(group), rng(10), offset(offset)
{
  // Generate groups of random samples
  sample_points.resize(NUM_SAMPLE_GROUPS, ptc->num_sample_divs);
  for(int sgindex=0;sgindex<NUM_SAMPLE_GROUPS;sgindex++) {
    for(int div_index=0; div_index < ptc->num_sample_divs; div_index++) {
      double inc = 1./ptc->num_samples_root[div_index];
      sample_points(sgindex, div_index).resize(ptc->num_samples[div_index]);
      
      // This is our sample index
      int index=0;
      // u and v are offsets into the texel
      double u=0;
      for (int i=0;i<ptc->num_samples_root[div_index];i++) {
	ASSERT(index<ptc->num_samples[div_index]);
	double v=0;
	for (int j=0;j<ptc->num_samples_root[div_index];j++) {
	  sample_points(sgindex, div_index)[index] = Point2D(u + inc*rng(),
							     v + inc*rng());
	  
	  index++;
	  v+=inc;
	}
	u+=inc;
      }
#if 1
      // Reshuffle our points, so that we don't get any correlation
      // between sets of points.
      for(int sample = 0; sample < ptc->num_samples[div_index]; sample++) {
	Point2D temp = sample_points(sgindex, div_index)[sample];
	int random_index =
          static_cast<int>(rng() * (ptc->num_samples[div_index]-1));
	sample_points(sgindex, div_index)[sample] = 
	  sample_points(sgindex, div_index)[random_index];
	sample_points(sgindex, div_index)[random_index] = temp;
      }
#endif
    } // end div_index
  } // end sgindex

  // Allocate the DepthStats and PerProcessor context
  ppc = new PerProcessorContext(ptc->pp_size, ptc->pp_scratchsize);
  depth_stats = new DepthStats();

  // Copy base filename
  basename=strdup(texname);
}

PathTraceWorker::~PathTraceWorker() {
//  cerr << "PathTraceWorker::~PathTraceWorker()\n";
  if (ppc) delete ppc;
  if (depth_stats) delete depth_stats;
  if (basename) free(basename);
  // Delete all the geometry used to store the texture data
  for(int i = 0; i < local_spheres->objs.size(); i++)
    if (local_spheres->objs[i])
      delete local_spheres->objs[i];
}

void PathTraceWorker::run() {
  Array1<float> inv_num_samples(ptc->num_sample_divs);
  for (int i = 0; i < inv_num_samples.size(); i++) {
    inv_num_samples[i] = 1./ptc->num_samples[i];
  }

  int* reflect_sample_set = new int[ptc->max_depth];
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

    for(int v=0;v<height;v++) {
      // Compute which number of samples to use
      float norm = v*2.0f/(height-1) - 1;
      float index_thing = norm * (ptc->num_sample_divs-1);
      if (index_thing < 0)
	index_thing *= -1;
      int div_index = (int)(index_thing+0.4f);
      
      for(int u=0;u<width;u++) {
	int sgindex = (int)(rng()*(NUM_SAMPLE_GROUPS-1));
        for(int rss = 0; rss < ptc->max_depth; rss++)
          reflect_sample_set[rss] = (int)(rng()*(NUM_SAMPLE_GROUPS-1));
#if 0
	sphere->texture(u,v)+=div_index;
	continue;
#endif

        for(int sample=0;sample<ptc->num_samples[div_index];sample++) {
          Point2D sample_point = sample_points(sgindex, div_index)[sample];
	  
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

	  float result=0;
	  PathTraceLight* light=&(ptc->light);
	  for (int depth=0;depth<=ptc->max_depth;depth++) {
            if (ptc->compute_directlighting || ptc->compute_shadows) {
#if 0
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
#else
              
              double light_norm_dot_sr = 1;
              Vector sr_dir=light->center-origin;
#endif
              
              double distance=sr_dir.normalize();
              double normal_dot_sr=SCIRun::Dot(sr_dir, normal);
              if(normal_dot_sr>0.0) {
                Ray s_ray(origin, sr_dir);
                HitInfo s_hit;

                if (ptc->compute_shadows) {
                  s_hit.min_t=distance;
                  Color s_color(1,1,1);
                  
                  ptc->geometry->light_intersect(s_ray, s_hit, s_color,
                                                 depth_stats, ppc);
                }
                
                if (!s_hit.was_hit)
                  result +=
                    ptc->luminance * light->luminance *
                    light_norm_dot_sr *
                    light->area *
                    (normal_dot_sr/(distance*distance*M_PI));
              }
            } // end if (compute_shadows || compute_directlighting)
            
	    // Pick a random direction on the hemisphere
	    Vector v0(Cross(normal, Vector(1,0,0)));
	    if(v0.length2()==0)
	      v0=Cross(normal, Vector(0,1,0));
	    v0.normalize();
	    Vector v1=Cross(normal, v0);
	    v1.normalize();
	    
	    Vector v_out;
	    {
	      Point2D hemi_sample=sample_points(reflect_sample_set[depth],
                                                div_index)[sample];
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

		sphere->inside(u,v) += 1;
		break;
	      }
	    }
	    else {
	      // Accumulate bg color?
	      Color bgcolor;
	      ptc->background->color_in_direction(ray.direction(), bgcolor);
	      result += bgcolor.luminance()*ptc->luminance;
	      break;
	    }
	  } // end depth
	  
          // Store result
	  sphere->texture(u,v)+=result;
	} // end sample
	
        // Normalize result
	sphere->texture(u,v)=inv_num_samples[div_index]*sphere->texture(u,v);
      } // end column of texels
    } // end all texels
    // cout << "Finished sphere "<<sindex<<"\n";
  } // end sphere

  // cout << "Computing width and height for all.\n";
  int width = -1;
  int height = -1;
  bool do_separate = false;
  // Write out textures
  for(int sindex = 0; sindex < local_spheres->objs.size(); sindex++) {
    TextureSphere *sphere  =
      dynamic_cast<TextureSphere*>(local_spheres->objs[sindex]);
    if (!sphere) {
      // Something has gone horribly wrong
      continue;
    }
    if (width > 0) {
      if (width != sphere->texture.dim1()) {
	do_separate = true;
	break;
      }
    } else {
      width = sphere->texture.dim1();
    }
    if (height > 0) {
      if (height != sphere->texture.dim2()) {
	do_separate = true;
	break;
      }
    } else {
      height = sphere->texture.dim2();
    }
  }

  if (!do_separate) {
    char *buf = new char[strlen(basename) + 20];
    sprintf(buf, "%s%07lu.nrrd", basename, offset);
    FILE *out = fopen(buf, "wb");
    if (!out) {
      cerr << "Cannot open "<<buf<<" for writing\n";
      return;
    }

    fprintf(out, "NRRD0001\n");
    fprintf(out, "type: float\n");
    fprintf(out, "dimension: 3\n");
    fprintf(out, "sizes: %d %d %d\n", width, height, local_spheres->objs.size());
    fprintf(out, "spacings: 1 1 NaN\n");
#ifdef __sgi
    fprintf(out, "endian: big\n");
#else
    fprintf(out, "endian: little\n");
#endif
    fprintf(out, "labels: \"x\" \"y\" \"sphere\"\n");
    fprintf(out, "encoding: raw\n");
    fprintf(out, "\n");
    
    for(int sindex = 0; sindex < local_spheres->objs.size(); sindex++) {
      TextureSphere *sphere  =
	dynamic_cast<TextureSphere*>(local_spheres->objs[sindex]);
      if (!sphere) {
	// Something has gone horribly wrong
	cerr << "Warning object is not a TextureSphere.  You in big trouble mister!  Nrrd will not have the right amount of data.\n";
	continue;
      }
      
      sphere->writeData(out, sindex, ptc);
    }
    
    fclose(out);

    cout << "Wrote combined texture to "<<buf<<"\n";
  } else {
    // This really shouldn't happen now
    cerr << "Textures do not all have the same size.  You will now get individual textures.\n";
    for(int sindex = 0; sindex < local_spheres->objs.size(); sindex++) {
      TextureSphere *sphere  =
	dynamic_cast<TextureSphere*>(local_spheres->objs[sindex]);
      if (!sphere) {
	// Something has gone horribly wrong
	cerr << "Warning object is not a TextureSphere.  You in big trouble mister!  Nrrd will not have the right amount of data.\n";
	continue;
      }
      sphere->writeTexture(basename, sindex, ptc);
    }
  }
  if (ptc->sem)
    ptc->sem->up();
}

TextureSphere::TextureSphere(const Point &cen, double radius, int tex_res):
  Sphere(0, cen, radius), texture(tex_res, tex_res), inside(tex_res, tex_res)
{
  texture.initialize(0);
  inside.initialize(0);
}

void TextureSphere::dilateTexture(size_t /*index*/, PathTraceContext* ptc) {
  //  cout<<"Dilating texture "<<index<<" ("<<"s="<<ptc->support
  //      <<", uwa="<<ptc->use_weighted_ave<<", t="<<ptc->threshold<<")"
  //      <<endl;
  
  // Compute the min and max of inside texture
  int width = inside.dim1();
  int height = inside.dim2();
  float min=inside(0,0);
  float max=inside(0,0);
  for (int y=1;y<height;y++)
    for (int x=1;x<width;x++) {
      float tmp=inside(x,y);
      if (tmp < min)
	min = tmp;
      if (tmp > max)
	max = tmp;
    }
  
  // Normalize the inside texture
  float inv_maxmin = 1/(max-min);
  for (int y=0;y<height;y++)
    for (int x=0;x<width;x++)
      inside(x,y)=(inside(x,y)-min)*inv_maxmin;

  // Initialize the dilated texture
  Array2<float> dilated;
  dilated.resize(width, height);
  for (int y=0;y<height;y++)
    for (int x=0;x<width;x++)
      dilated(x,y)=texture(x,y);
  
  // Dilate any necessary pixels
  int support=ptc->support;
  int use_weighted_ave=ptc->use_weighted_ave;
  float threshold=ptc->threshold;
  
  for(int y = 0; y < height; y++)
    for(int x = 0; x < width; x++)
      {
	// Determine if the given pixel should be dilated
	float value=inside(x,y);
	if (value<=0)
	  // Pixel is not occluded, so go to the next one
	  continue;
	
	float ave = 0;
	float contribution_total = 0;
	// Loop over each neighbor
	for(int j = y-support; j <= y+support; j++)
	  for(int i = x-support; i <= x+support; i++)
	    {
	      // Check boundary conditions
	      int newi = i;
	      if (newi >= width)
		newi = newi - width;
	      else if (newi < 0)
		newi += width;
	      
	      int newj = j;
	      if (newj >= height)
		newj = height - 1;
	      else if (newj < 0)
		newj = 0;

	      // Determine neighbor's contribution
	      float contributer=inside(newi, newj);
	      if (contributer < threshold) {
		contributer*=use_weighted_ave;
		ave+=texture(newi, newj)*(1-contributer);
		contribution_total+=(1-contributer);
	      }
	    }
	
	// Dilate the pixel
	if (contribution_total > 0) {
	  dilated(x,y)=ave/contribution_total;
	}
      }
  
  // Update texture with dilated results
  for (int y=0;y<height;y++)
    for (int x=0;x<width;x++)
      texture(x,y)=dilated(x,y);
}

void TextureSphere::writeTexture(char* basename, size_t index,
				 PathTraceContext* ptc)
{
  // Create the filename
  char *buf = new char[strlen(basename) + 20];
  sprintf(buf, "%s%07lu.nrrd", basename, (unsigned long)index);
  FILE *out = fopen(buf, "wb");
  if (!out) {
    cerr << "Cannot open "<<buf<<" for writing\n";
    return;
  }
  
  // write out texutre
  int width = texture.dim1();
  int height = texture.dim2();
  fprintf(out, "NRRD0001\n");
  fprintf(out, "type: float\n");
  fprintf(out, "dimension: 2\n");
  fprintf(out, "sizes: %d %d\n", width, height);
  fprintf(out, "spacings: 1 1\n");
#ifdef __sgi
  fprintf(out, "endian: big\n");
#else
  fprintf(out, "endian: little\n");
#endif
  fprintf(out, "labels: \"x\" \"y\"\n");
  fprintf(out, "encoding: raw\n");
  fprintf(out, "\n");
  
  writeData(out, index, ptc);
  
  fclose(out);
}

void TextureSphere::writeData(FILE *outfile, size_t index, PathTraceContext* ptc) {
  // Dilate the texture
  if (ptc->dilate)
    dilateTexture(index, ptc);
  
  // Iterate over each texel
  int width = texture.dim1();
  int height = texture.dim2();
  for(int v = 0; v < height; v++) {
    for(int u = 0; u < width; u++) {
      float data = texture(u,v);
      if (fwrite(&data, sizeof(float), 1, outfile) != 1) {
	cerr << "Trouble writing texel for sphere at ["<<u<<", "<<v<<"]\n";
	return;
      }
    }
  }
}

PathTraceLight::PathTraceLight(const Point &cen, double radius,
			       float lum):
  center(cen), radius(radius), luminance(lum)
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
