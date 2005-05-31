
#ifndef SKETCHMATERIAL_H
#define SKETCHMATERIAL_H 1

#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/SketchMaterialBase.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

#include <Core/Geometry/Point.h>
#include <Core/Math/Expon.h>

#include <teem/nrrd.h>
#include <teem/gage.h>
#include <stdlib.h>

extern "C"
int _gageLocationSet (gageContext *ctx, gage_t x, gage_t y, gage_t z);

//#define COMPUTE_K1_K2 1

namespace rtrt {

template<class ArrayType, class DataType>  
class SketchMaterial : public SketchMaterialBase, public Material {
protected:
  ArrayType data;
  int nx, ny, nz;
  BBox bbox;
  Vector inv_diag;
  float ambient, specular, diffuse;
  int spec_coeff;
  // This is the array used for the silhouette edge lookup
  Array2<float> sil_trans_funct;

  gageContext *main_ctx;
  gagePerVolume *pvl;

  // This is created in the constructor and should have as many
  // contextes as there are processors.  Indexing into here should be
  // made by the worker num.
  Array1<gageContext*> ctx_pool;

  // This is used to do cool to warm colormaps
  ScalarTransform1D<float, Color> *cool2warm;
  // This is used for the shadow color when using cool2warm
  Color shadow_color;

  Color color(const Vector &N, const Vector &V, const Vector &L, 
	      const Color &object_color, const Color &light_color) const;
  int rtrtGageProbe(gageContext *ctx, gage_t x, gage_t y, gage_t z);
public:
  SketchMaterial(ArrayType &indata, BBox &bbox, Array2<float>& sil_trans,
		 float sil_thickness, ScalarTransform1D<float, Color>* cm);
  virtual ~SketchMaterial();
  
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);

  virtual void animate(double t, bool& changed);
};

template<class ArrayType, class DataType>
SketchMaterial<ArrayType, DataType>::SketchMaterial
(ArrayType &indata, BBox &bbox, Array2<float>& sil_trans, float sil_thickness,
 ScalarTransform1D<float, Color> *cm):
  SketchMaterialBase(sil_thickness),
  bbox(bbox), ambient(0.5f), specular(1), diffuse(1), spec_coeff(32),
  cool2warm(cm)
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

  // Set up the silhouette transfer function
  sil_trans_funct.share(sil_trans);
  
  // Set up the fake nrrd we can use to setup the gage stuff
  Nrrd *nin = nrrdNew();
  nrrdWrap(nin, data.get_dataptr(), nrrdTypeShort, 3, nx, ny, nz);
  // Setup the spacing for the data.  This assums node centered data,
  // which is what rtrt uses.
  nin->axis[0].spacing = diag.x()/(nx-1);
  nin->axis[1].spacing = diag.y()/(ny-1);
  nin->axis[2].spacing = diag.z()/(nz-1);

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
  gageQuery query;
  GAGE_QUERY_RESET(query);
  GAGE_QUERY_ITEM_ON(query, gageSclGradVec);
  GAGE_QUERY_ITEM_ON(query, gageSclGeomTens);
#ifdef COMPUTE_K1_K2
  GAGE_QUERY_ITEM_ON(query, gageSclK1);
  GAGE_QUERY_ITEM_ON(query, gageSclK2);
#endif
  
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

  //  int fd = GAGE_FD(main_ctx);
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
  // check with the thread library.  Add one more for an auxiliary
  // thread to use.
  ctx_pool.resize(SCIRun::Thread::numProcessors()+1);
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

  // This will set up the colormap properly
  shadow_color = cool2warm->lookup(0);
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
                                           double /*atten*/,
                                           const Color& /*accumcolor*/,
                                           Context* cx) {
  Point hit_pos(ray.origin()+ray.direction()*(hit.min_t));
  if (normal_method == 0) {
    Color surface(0.9, 0.9, 0.9);
    // Compute whether we are in shadow
    Color shadowfactor(1,1,1);
    Light* light=cx->scene->light(0);
    Vector light_dir;
    light_dir = light->get_pos()-hit_pos;
    double dist=light_dir.normalize();
    if(cx->scene->lit(hit_pos, light, light_dir, dist, shadowfactor, depth, cx) ){
      // Not in shadow

      // Get the normal from the object
      Vector normal(hit.hit_obj->normal(hit_pos, hit));
      
      result = color(normal, ray.direction(), light_dir, 
		     surface, light->get_color());
    } else {
      // we are in shadow
      result = light->get_color() * surface * ambient;
    }
    return;
  }
  if (bbox.contains_point(hit_pos)) {
    // Get the ctx for this worker
    int rank = cx->worker_num;
    if (rank == -1) rank = ctx_pool.size()-1;
    if (rank >= ctx_pool.size() || rank < -1) {
      // The rank is bad, so lets use some generic color.
      result = Color(1,0,1);
      return;
    }

    // The rank will work
    gageContext *gctx = ctx_pool[rank];
    
    // Compute the point in index space
    Vector norm_hit_pos(hit_pos - bbox.min());
    
    double idx_norm = norm_hit_pos.x() * inv_diag.x();
    double samplex = idx_norm * (nx - 1);
    
    idx_norm = norm_hit_pos.y() * inv_diag.y();
    double sampley = idx_norm * (ny - 1);
    
    idx_norm = norm_hit_pos.z() * inv_diag.z();
    double samplez = idx_norm * (nz - 1);
    
    if (rtrtGageProbe(gctx, samplex, sampley, samplez) != 0) {
      // There was a problem
      result = Color(1,1,0);
      return;
    }
    
#ifdef COMPUTE_K1_K2
    gage_t *k1 = gageAnswerPointer(gctx, gctx->pvl[0], gageSclK1);
    gage_t *k2 = gageAnswerPointer(gctx, gctx->pvl[0], gageSclK2);
#endif
    gage_t *norm = gageAnswerPointer(gctx, gctx->pvl[0], gageSclGradVec);
    gage_t *geomt = gageAnswerPointer(gctx, gctx->pvl[0], gageSclGeomTens);
    
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
    
    Color surface(0.9, 0.9, 0.9);
    // Compute whether we are in shadow
    Color shadowfactor(1,1,1);
    Vector light_dir;
    light_dir = light->get_pos()-hit_pos;
    double dist=light_dir.normalize();
    if(cx->scene->lit(hit_pos, light, light_dir, dist, shadowfactor, depth, cx) ){
      // Not in shadow
      if (use_cool2warm) {
        double dot = Dot(light_dir, normal);
        surface = cool2warm->lookup_bound(dot);
      } else {
        surface = color(normal, ray.direction(), light_dir, 
                        surface, light->get_color());
      }
    } else {
      // we are in shadow
      if (use_cool2warm) {
        double dot = Dot(light_dir, normal);
        if (dot >= 0)
          surface = cool2warm->lookup_bound(dot);
        else
          // This is regions facing the light.
          surface = cool2warm->lookup_bound(dot*0.25);
        //	      surface = cool2warm->lookup_bound(-dot);
      } else {
        surface = light->get_color() * surface * ambient;
      }
    }
    
    if (show_silhouettes == 0) {
      result = surface;
      return;
    }
    // Cool, now let's lookup the silhouette contribution.
    
    // We need a dot product of the normal with the view vector.
    // This should be between -1 and 1;
    Vector view = -ray.direction().normal();
    double eye_dot_norm = Dot(view, normal);
    
    // Now the multiplication of the geom tensor with the view.
    double viewx = view.x();
    double viewy = view.y();
    double viewz = view.z();
    double eye_gt_eye =
      viewx*(viewx * geomt[0] + viewy * geomt[1] + viewz * geomt[2]) +
      viewy*(viewx * geomt[3] + viewy * geomt[4] + viewz * geomt[5]) +
      viewz*(viewx * geomt[6] + viewy * geomt[7] + viewz * geomt[8]);
    
    // Now to compute the indecies for the lookup, with bounds checks
    int silx = static_cast<int>((eye_dot_norm + 1) * 0.5 *
                                (sil_trans_funct.dim1()-1));
    int sily = static_cast<int>(eye_gt_eye * inv_sil_thickness *
                                (sil_trans_funct.dim2()-1));
    //	cerr << "eye_dot_norm = "<<eye_dot_norm<<", eye_gt_eye = "<<eye_gt_eye<<", silx,y = ("<<silx<<", "<<sily<<")\n";cerr.flush();
    if (silx < 0 || silx >= sil_trans_funct.dim1())
      silx = 0;
    if (sily < 0 || sily >= sil_trans_funct.dim2())
      sily = 0;
    
    // Bounds checks on sil_val???
    float sil_val = sil_trans_funct(silx, sily);
    // Now to do a lerp
    result = surface * sil_val + sil_color * (1 - sil_val);
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
    spec = attenuation * specular * SCIRun::Pow(exponent, spec_coeff);
  } else {
    spec = attenuation * specular * SCIRun::Pow(-exponent, spec_coeff);
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
    double spec = attenuation * specular * SCIRun::Pow(Max(Dot(R, V),0.0), spec_coeff);

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

template<class ArrayType, class DataType>
void
SketchMaterial<ArrayType, DataType>::animate(double /*t*/, bool& changed) {
  // Here we can update all the gage stuff if we need to.
  if (gui_sil_thickness != sil_thickness ||
      gui_use_cool2warm != use_cool2warm ||
      gui_sil_color_r != sil_color.red() ||
      gui_sil_color_g != sil_color.green() ||
      gui_sil_color_b != sil_color.blue() ||
      gui_show_silhouettes != show_silhouettes ||
      gui_normal_method != normal_method)
    {
      // Update the thickness
      sil_thickness = gui_sil_thickness;
      if (gui_sil_thickness != 0)
	inv_sil_thickness = 1/gui_sil_thickness;
      else
	inv_sil_thickness = 0;
      // Update the silhouette color
      sil_color = Color(gui_sil_color_r, gui_sil_color_g, gui_sil_color_b);
      // Update show_silhouettes
      show_silhouettes = gui_show_silhouettes;
      // Update normal_method
      normal_method = gui_normal_method;

      use_cool2warm = gui_use_cool2warm;
      
      changed = true;
    }
}

} // end namespace rtrt

#endif
