
#ifndef SKETCHMATERIAL_H
#define SKETCHMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>
#include <teem/nrrd.h>
#include <teem/gage.h>
#include <stdlib.h>

extern "C"
int _gageLocationSet (gageContext *ctx, gage_t x, gage_t y, gage_t z);

namespace rtrt {

template<class ArrayType, class DataType>  
class SketchMaterial : public Material {
protected:
  ArrayType data;
  int nx, ny, nz;
  BBox bbox;
  Vector inv_diag;
  float ambient, specular, spec_coeff, diffuse;

  gageContext *main_ctx;
  gagePerVolume *pvl;

  // This is created in the constructor and should have as many
  // contextes as there are processors.  Indexing into here should be
  // made by the worker num.
  Array1<gageContext*> ctx_pool;

  Color color(const Vector &N, const Vector &V, const Vector &L, 
	      const Color &object_color, const Color &light_color) const;
public:
  SketchMaterial(ArrayType &indata, BBox &bbox);
  virtual ~SketchMaterial();
  
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);
  int rtrtGageProbe(gageContext *ctx, gage_t x, gage_t y, gage_t z);
};

template<class ArrayType, class DataType>
SketchMaterial<ArrayType, DataType>::SketchMaterial(ArrayType &indata, BBox &bbox):
  bbox(bbox), ambient(0.5f), specular(1), spec_coeff(64), diffuse(1)
{
  char me[] = "SketchMaterial::SketchMaterial";
  char *errS;
  // Share the data coming in
  data.share(indata);
  nx = data.dim1();
  ny = data.dim2();
  nz = data.dim3();

  // Set up the some of the geometry
  Vector diag(bbox.max() - bbox.min());
  if (diag.x() == 0)
    inv_diag.x(0);
  else
    inv_diag.x(1.0/diag.x());

  if (diag.y() == 0)
    inv_diag.y(0);
  else
    inv_diag.y(1.0/diag.y());

  if (diag.z() == 0)
    inv_diag.z(0);
  else
    inv_diag.z(1.0/diag.z());
  
  // Set up the fake nrrd we can use to setup the gage stuff
  Nrrd *nin = nrrdNew();
  nrrdWrap(nin, data.get_dataptr(), nrrdTypeShort, 3, nx, ny, nz);
  // We need default spacings for the data.
  // Should this be computed based on the physical size of the volume?
#if 0
  nrrdAxisSpacingSet(nin, 0);
  nrrdAxisSpacingSet(nin, 1);
  nrrdAxisSpacingSet(nin, 2);
#else
  // This assums node centered data, which is what rtrt uses.
  nin->axis[0].spacing = diag.x()/(nx-1);
  nin->axis[1].spacing = diag.y()/(ny-1);
  nin->axis[2].spacing = diag.z()/(nz-1);
#endif

  //  fprintf(stderr, "%s: sizeof(gageContext) = %x (%d)\n", me, sizeof(gageContext), sizeof(gageContext));
  //  fprintf(stderr, "%s: sizeof(gagePerVolume) = %x (%d)\n", me, sizeof(gagePerVolume), sizeof(gagePerVolume));
  
  
  main_ctx = gageContextNew();
  if (!main_ctx) {
    fprintf(stderr, "%s: gageContextNew failed\n", me);
    return;
  }
  //  fprintf(stderr, "%s: main_ctx created with pointer: %x (%lu)\n", me, main_ctx, (unsigned long)main_ctx);
  
  pvl = gagePerVolumeNew(main_ctx, nin, gageKindScl);
  if (!pvl) {
    fprintf(stderr, "%s: gagePerVolumeNew failed\n", me);
    return;
  }
  //  fprintf(stderr, "%s: pvl created with pointer: %x (%lu)\n", me, pvl, (unsigned long)pvl);
  
  if (gagePerVolumeAttach(main_ctx, pvl)) {
    fprintf(stderr, "%s: problem with gagePerVolumeAttach:\n%s\n",
	    me, errS = biffGetDone(GAGE));
    free(errS);
    return;
  }
  //  fprintf(stderr, "%s: gagePerVolumeAttach finished without errors\n", me);
  
  // Set up what you are querying for
  unsigned int query = 0;
  query = (1<<gageSclK1) | (1<<gageSclK2) | (1<<gageSclGradVec) |
    (1<<gageSclGeomTens);
  gageQuerySet(main_ctx, pvl, query);
  
  // Set up the convolution kernels
  
  // Scalar Reconstruction
  NrrdKernelSpec *ksp0 = nrrdKernelSpecNew();
  nrrdKernelParse(&(ksp0->kernel), ksp0->parm, "cubic:1,1,0");
  int E = gageKernelSet(main_ctx, gageKernel00, ksp0->kernel, ksp0->parm);
  // First Derivative
  NrrdKernelSpec *ksp1 = nrrdKernelSpecNew();
  nrrdKernelParse(&(ksp1->kernel), ksp1->parm, "cubicd:1,1,0");
  if (!E) E = gageKernelSet(main_ctx, gageKernel11, ksp1->kernel, ksp1->parm);
  // Second Derivative
  NrrdKernelSpec *ksp2 = nrrdKernelSpecNew();
  nrrdKernelParse(&(ksp2->kernel), ksp2->parm, "cubicdd:1,1,0");
  if (!E) E = gageKernelSet(main_ctx, gageKernel22, ksp2->kernel, ksp2->parm);
  if (E) {
    fprintf(stderr, "%s: problem with setting up the kernels:\n%s\n",
	    me, errS = biffGetDone(GAGE));
    free(errS);
    return;
  }

  int fd = GAGE_FD(main_ctx);
  //  fprintf();
  
  //  fprintf(stderr, "%s: gageKernelSet for all finished without errors\n", me);

  if (gageUpdate(main_ctx)) {
    fprintf(stderr, "%s: problem with gageUpdate:\n%s\n",
	    me, errS = biffGetDone(GAGE));
    free(errS);
    return;
  }
  fprintf(stderr, "%s: gageUpdate finished without errors\n", me);
  
  // Remove the padded data
  if (pvl->nixer && pvl->npad) {
    pvl->nixer(pvl->npad, pvl->kind, pvl);
    pvl->npad = NULL;
  }
  
  // Now we need to make duplications of the gageContext, so each
  // thread can use their own.  For now I will not worry about cache
  // locality and worry about getting the rest of the code correct.
  
  // We need to know how many threads we could possibly use.  Lets
  // check with the thread library.
  ctx_pool.resize(Thread::numProcessors());
  fprintf(stderr, "%s: ctx_pool resized to %d\n", me, ctx_pool.size());
  for(int i = 0; i< ctx_pool.size(); i++) {
    // Allocate a new ctx
    ctx_pool[i] = gageContextCopy(main_ctx);
    if (ctx_pool[i] == 0) {
      // There was a problem
      fprintf(stderr, "%s: problem allocating ctx_pool[%d]: %s\n", me, i,
	      errS = biffGetDone(GAGE));
      free(errS);
      return;
    }
  }
  fprintf(stderr, "%s: ctx_pool finished allocating\n", me);
  
  
  // Should I nix nin?
  // nrrdNix(nin);
  // What about the kernel specs?
  // nrrdKernelSpecNix(ksp0);
  // nrrdKernelSpecNix(ksp1);
  // nrrdKernelSpecNix(ksp2);
}
  
template<class ArrayType, class DataType>
SketchMaterial<ArrayType, DataType>::~SketchMaterial() {
  // I should free all the gageContexts
  for(int i = 0; i< ctx_pool.size(); i++) {
    if (ctx_pool[i])
      gageContextNix(ctx_pool[i]);
  }
  gageContextNix(main_ctx);
  gagePerVolumeNix(pvl);
}
  
template<class ArrayType, class DataType>
void
SketchMaterial<ArrayType, DataType>::shade(Color& result, const Ray& ray,
					const HitInfo& hit, int depth,
					double atten, const Color& accumcolor,
					Context* cx) {
  Point hit_pos(ray.origin()+ray.direction()*(hit.min_t));
  if (bbox.contains_point(hit_pos)) {
    // Get the ctx for this worker
    int rank = cx->worker->rank();
    if (rank < ctx_pool.size() && rank >= 0) {
      // The rank will work
      gageContext *gctx = ctx_pool[rank];
      
      // Compute the point in index space
      Vector norm_hit_pos(hit_pos - bbox.min());
      
      double norm = norm_hit_pos.x() * inv_diag.x();
      double samplex = norm * (nx - 1);
      
      norm = norm_hit_pos.y() * inv_diag.y();
      double sampley = norm * (ny - 1);
      
      norm = norm_hit_pos.z() * inv_diag.z();
      double samplez = norm * (nz - 1);
      
      if (rtrtGageProbe(gctx, samplex, sampley, samplez) == 0) {
	gage_t *k1 = gageAnswerPointer(gctx, gctx->pvl[0], gageSclK1);
	gage_t *k2 = gageAnswerPointer(gctx, gctx->pvl[0], gageSclK2);
	gage_t *norm = gageAnswerPointer(gctx, gctx->pvl[0], gageSclGradVec);
	
	//      printf("k1 = %g, k2 = %g, norm = [%g, %g, %g]\n",*k1, *k2,
	//	     norm[0], norm[1], norm[2]);
	Vector normal;
	gage_t length2 = norm[0]*norm[0]+norm[1]*norm[1]+norm[2]*norm[2];
	if (length2){
	  // this lets the compiler use a special 1/sqrt() operation
	  float ilength2 = 1.0f/sqrtf(length2);
	  normal = Vector(norm[0]*ilength2, norm[1]*ilength2, norm[2]*ilength2);
	} else {
	  normal = Vector(0,0,0);
	}
	
	Light* light=cx->scene->light(0);
	Vector light_dir;
	light_dir = light->get_pos()-hit_pos;
	
	result = color(normal, ray.direction(), light_dir.normal(), 
		       Color(1,0.3,0.2), light->get_color());
      } else {
	return;
      }
    } else {
      // The rank is bad, so lets use some generic color.
      result = Color(1,0,1);
    }
  }
}

template<class ArrayType, class DataType>
int
SketchMaterial<ArrayType, DataType>::rtrtGageProbe(gageContext *ctx,
						gage_t x, gage_t y, gage_t z)
{
  //  char me[]="rtrtgageProbe";
  
  if (_gageLocationSet(ctx, x, y, z)) {
    /* we're outside the volume; leave gageErrStr and gageErrNum set
       (as they should be) */
    return 1;
  } 
    
  for (int i=0; i<ctx->numPvl; i++) {
    // Need to copy the data over to the iv3 struct in pvl
    gage_t *iv3 = ctx->pvl[i]->iv3;
    // This is the filter diameter.  You will have to copy fd*fd*fd
    // values into iv3.
    int fd = GAGE_FD(ctx);
    // These are the indicies which will start the copying.  You
    // should do bounds checking on these indicies as they will be out
    // of bounds at the boundary.
    int startx, starty, startz;
    // point.xi ect. are in the padded space.  Subtract havePad to get
    // back to UNPADDED space.  Then again to get the start for the
    // kernel support.
    startx = ctx->point.xi - 2*ctx->havePad;
    starty = ctx->point.yi - 2*ctx->havePad;
    startz = ctx->point.zi - 2*ctx->havePad;
    //    printf("startx,y,z = %d, %d, %d, ctx->havePad = %d\n", startx, starty, startz, ctx->havePad);
    int nxm1 = nx - 1;
    int nym1 = ny - 1;
    int nzm1 = nz - 1;
    for(int z = startz; z < startz+fd; z++) {
      int zindex = AIR_CLAMP(0, z, nzm1);
      for(int y = starty; y < starty+fd; y++) {
	int yindex = AIR_CLAMP(0, y, nym1);
	for(int x = startx; x < startx+fd; x++) {
	  int xindex = AIR_CLAMP(0, x, nxm1);
	  *iv3++ = data(xindex, yindex, zindex);
	}
      }
    }
  } // end for each pvl

  /* fprintf(stderr, "##%s: bingo 2\n", me); */
  for (int i=0; i<ctx->numPvl; i++) {
#if 0
    if (ctx->verbose > 1) {
      fprintf(stderr, "%s: pvl[%d]'s value cache with (unpadded) "
	      "coords = %d,%d,%d:\n", me, i,
	      ctx->point.xi - ctx->havePad,
	      ctx->point.yi - ctx->havePad,
	      ctx->point.zi - ctx->havePad);
      ctx->pvl[i]->kind->iv3Print(stderr, ctx, ctx->pvl[i]);
    }
#endif
    ctx->pvl[i]->kind->filter(ctx, ctx->pvl[i]);
    ctx->pvl[i]->kind->answer(ctx, ctx->pvl[i]);
  }
  
  /* fprintf(stderr, "##%s: bingo 5\n", me); */
  return 0;
}
  
template<class ArrayType, class DataType>
Color
SketchMaterial<ArrayType, DataType>::color(const Vector &N, const Vector &V,
				 const Vector &L, const Color &object_color,
				 const Color &light_color) const {

  Color result; // the resulting color

  double L_N_dot = Dot(L, N);

#if 1 // Double Sided shading
  double attenuation = 1;
  Vector L_use;

  // the dot product is negative then the objects face points
  // away from the light and the normal should be reversed.
  if (L_N_dot >= 0) {
    L_use = L;
  } else {
    L_N_dot = -L_N_dot;
    L_use = -L;
  }

  // do the ambient, diffuse, and specular calculations
  double exponent;
#if 0 // Use Halfway vector instead of reflection vector
  //  Vector H = (L + V) * 0.5f;
  Vector H = (L_use + V) * 0.5f;
  exponent = Dot(N, H);
#else
  Vector R = N * (2.0 * L_N_dot) - L_use;
  exponent = Dot(R, V);
#endif
  double spec;
  if (exponent > 0) {
    spec = attenuation * specular * pow(exponent, spec_coeff*0.5);
  } else {
    spec = attenuation * specular * pow(-exponent, spec_coeff*0.5);
  }
  
  result = light_color * (object_color *(ambient+attenuation*diffuse*L_N_dot)
			  + Color(spec, spec, spec));
#else
  // the dot product is negative then the objects face points
  // away from the light and should only contribute an ambient term
  if (L_N_dot > 0) {
    // do the ambient, diffuse, and specular calculations
    double attenuation = 1;

    Vector R = N * (2.0 * L_N_dot) - L;
    double spec = attenuation * specular * pow(Max(Dot(R, V),0.0), spec_coeff*0.5);

    result = light_color * (object_color *(ambient+attenuation*diffuse*L_N_dot)
			    + Color(spec, spec, spec));
  }
  else {
    // do only the ambient calculations
    result = light_color * object_color * ambient;
  }
#endif
  
  return result;
}

} // end namespace rtrt

#endif
