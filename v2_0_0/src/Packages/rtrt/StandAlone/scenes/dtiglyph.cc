/*
  dtiglyph.cc

  Scene file for rendering DTI tensor data, via Gordon's glyph scheme.

  Authors:  James Bigler (bigler@cs.utah.edu)
            Gordon Kindlmann (gk@cs.utah.edu)  (all use of macros)
  Date: July 10, 2002

*/

#include <Packages/rtrt/Core/Camera.h>
//#include <Packages/rtrt/Core/CatmullRomSpline.h>
//#include <Packages/rtrt/Core/GridSpheres.h>
//#include <Packages/rtrt/Core/GridSpheresDpy.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
//#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Glyph.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Core/Geometry/Transform.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <values.h>
#include <vector.h>

#include <teem/nrrd.h>
#include <teem/hest.h>
#include <teem/air.h>
#include <teem/biff.h>
#include <teem/ten.h>

using namespace rtrt;
using namespace std;

#define USE_GLYPH_GROUP

int
dtiParseNrrd(void *ptr, char *str, char err[AIR_STRLEN_HUGE]) {
  char me[] = "dtiParseNrrd", *nerr;
  Nrrd **nrrdP;
  airArray *mop;
  
  if (!(ptr && str)) {
    sprintf(err, "%s: got NULL pointer", me);
    return 1;
  }
  nrrdP = (Nrrd **)ptr;
  mop = airMopInit();
  *nrrdP = nrrdNew();
  airMopAdd(mop, *nrrdP, (airMopper)nrrdNuke, airMopOnError);
  if (nrrdLoad(*nrrdP, str, 0)) {
    airMopAdd(mop, nerr = biffGetDone(NRRD), airFree, airMopOnError);
    if (strlen(nerr) > AIR_STRLEN_HUGE - 1)
      nerr[AIR_STRLEN_HUGE - 1] = '\0';
    strcpy(err, nerr);
    airMopError(mop);
    return 1;
  }
  if (!tenValidTensor(*nrrdP, nrrdTypeFloat, AIR_TRUE)) {
    sprintf(err, "%s: \"%s\" isn't a valid tensor volume", me, str);
    biffAdd(TEN, err);
    airMopAdd(mop, nerr = biffGetDone(TEN), airFree, airMopOnError);
    if (strlen(nerr) > AIR_STRLEN_HUGE - 1)
      nerr[AIR_STRLEN_HUGE - 1] = '\0';
    strcpy(err, nerr);
    airMopError(mop);
    return 1;
  }
  airMopOkay(mop);
  return 0;
}

hestCB dtiNrrdHestCB = {
  sizeof(Nrrd *),
  "nrrd",
  dtiParseNrrd,
  (airMopper)nrrdNuke
};

int
dtiParseAniso(void *ptr, char *str, char err[AIR_STRLEN_HUGE]) {
  char me[]="dtiParseAniso";
  int *anisoP;
    
  anisoP = (int*)ptr;
  *anisoP = airEnumVal(tenAniso, str);
  if (tenAnisoUnknown == *anisoP) {
    sprintf(err, "%s: \"%s\" not a recognized anisotropy type", me, str);
    return 1;
  }
  return 0;
}

hestCB dtiAnisoHestCB = {
  sizeof(int),
  "aniso",
  dtiParseAniso,
  NULL
};

char *dtiINFO = ("Generates an rtrt scene to do glyph-based "
		 "visualization of a diffusiont-tensor field");

void
dtiRgbGen(Color &rgb, float evec[3], float an) {
  float r, g, b;

  r = AIR_ABS(evec[0]);
  g = AIR_ABS(evec[1]);
  b = AIR_ABS(evec[2]);
  rgb = Color(AIR_AFFINE(0.0, an, 1.0, 0.5, r),
	      AIR_AFFINE(0.0, an, 1.0, 0.5, g),
	      AIR_AFFINE(0.0, an, 1.0, 0.5, b));
}

namespace rtrt {
  extern float glyph_threshold;
}

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  airArray *mop;
  hestOpt *opt = NULL;
  int anisoType;
  char *me, *err;
  float glyphScale, anisoThresh;
  Nrrd *nin;
#ifdef USE_GLYPH_GROUP
  int gridcellsize;
  int num_levels;
#endif

  hestOptAdd(&opt, NULL, "input", airTypeOther, 1, 1, &nin, NULL,
	     "input tensor volume, in nrrd format, with 7 floats per voxel.",
	     NULL, NULL, &dtiNrrdHestCB);
  hestOptAdd(&opt, NULL, "aniso", airTypeEnum, 1, 1, &anisoType, NULL,
	     "scalar anisotropy type for thresholding and color "
	     "saturation modulation. "
	     "Currently supported:\n "
	     "\b\bo \"cl\": Westin's linear\n "
	     "\b\bo \"cp\": Westin's planar\n "
	     "\b\bo \"ca\": Westin's linear + planar\n "
	     "\b\bo \"cs\": Westin's spherical (1-ca)\n "
	     "\b\bo \"ct\": GK's anisotropy type (cp/ca)\n "
	     "\b\bo \"ra\": Basser+Pierpaoli relative anisotropy\n "
	     "\b\bo \"fa\": Basser+Pierpaoli fractional anisotropy/sqrt(2)\n "
	     "\b\bo \"vf\": volume fraction = 1-(Bass,Pier volume ratio)",
	     NULL, tenAniso);
  hestOptAdd(&opt, NULL, "scale", airTypeFloat, 1, 1, &glyphScale, NULL,
	     "over-all glyph scaling");
  hestOptAdd(&opt, NULL, "thresh", airTypeFloat, 1, 1, &anisoThresh, NULL,
	     "anisotropy threshold for testing");
#ifdef USE_GLYPH_GROUP
  hestOptAdd(&opt, "-gridcellsize", "gridcellsize", airTypeInt, 0, 1,
	     &gridcellsize, "3",
	     "size of the grid cells to put around the GlyphGroup");
  hestOptAdd(&opt, "-nl", "num_levels", airTypeInt, 0, 1, &num_levels, "10",
	     "number of grid levels to use for optimizations");
#endif
  
  mop = airMopInit();
  airMopAdd(mop, opt, (airMopper)hestOptFree, airMopAlways);
  me = argv[0];
  if (argc < 5) {
    hestInfo(stderr, me, dtiINFO, NULL);
    hestUsage(stderr, opt, me, NULL);
    hestGlossary(stderr, opt, NULL);
    airMopError(mop);
    return NULL;
  }
  fprintf(stderr, "%s: reading input ... ", me); fflush(stderr);
  if (hestParse(opt, argc-1, argv+1, &err, NULL)) {
    fprintf(stderr, "%s: %s\n", me, err); free(err);
    hestUsage(stderr, opt, me, NULL);
    hestGlossary(stderr, opt, NULL);
    airMopError(mop);
    return NULL;
  }
  printf("hooray!\n\n");
  fprintf(stderr, "done\n");
  airMopAdd(mop, opt, (airMopper)hestParseFree, airMopAlways);

  fprintf(stderr, "%s: glyphScale = %g\n", me, glyphScale);
  fprintf(stderr, "%s: anisoType = %d\n", me, anisoType);
  fprintf(stderr, "%s: anisoThresh = %g\n", me, anisoThresh);
  fprintf(stderr, "%s: got dti volume dimensions %d x %d x %d\n", me,
	  nin->axis[1].size, nin->axis[2].size, nin->axis[3].size);
  fprintf(stderr, "%s: dti volume spacings %g x %g x %g\n", me,
	  nin->axis[1].spacing, nin->axis[2].spacing, nin->axis[3].spacing);

  glyph_threshold = anisoThresh;
  //////////////////////////////////////////////////////
  // add geometry to this :)
  Group *all = new Group();

  int sx, sy, sz,        // sizes along x,y,z axes
    xi, yi, zi,          // indices into x,y,z axes
    numGlyphs;
  float zs, ys, xs;      // spacings along x,y,z axes
  sx = nin->axis[1].size;
  sy = nin->axis[2].size;
  sz = nin->axis[3].size;
  xs = nin->axis[1].spacing;
  ys = nin->axis[2].spacing;
  zs = nin->axis[3].spacing;
  float tmp,             // don't ask
    *tdata,              // all tensor data; 7 floats per tensor
    x, y, z,             // world-ish position (scaled by spacings)
    eval[3], evec[9],    // eigen{values,vectors} of tensor
    c[TEN_ANISO_MAX+1];  // all possible anisotropies
  tdata = (float*)nin->data;
  numGlyphs = 0;
#ifdef USE_GLYPH_GROUP
  Array1<Glyph*> glyphs;
#endif
  for (zi = 0; zi < sz; zi++) {
    z = zs * zi;
    for (yi = 0; yi < sy; yi++) {
      y = ys * yi;
      for (xi = 0; xi < sx; xi++, tdata+=7) {
	x = xs * xi;

	// we always ignore data points with confidence < 0.5
	if (!( tdata[0] > 0.4 ))
	  continue;

	// do eigensystem solve
	tenEigensolve(eval, evec, tdata);
	tenAnisoCalc(c, eval);
	//	if (!( c[anisoType] > anisoThresh))
	//	  continue;
	
	// so there will be a glyph generated for this sample
	numGlyphs++;
	Color rgb;
	dtiRgbGen(rgb, evec, c[anisoType]);
	Phong *matl = new Phong(rgb, Color(1,1,1), 100);
	// These are cool transparent/reflective glyphs
	//PhongMaterial *matl = new PhongMaterial(rgb, 0.3, 0.4, 100, true);
	// all glyphs start at the origin

	Sphere *obj = new Sphere(matl, Point(0,0,0), glyphScale);


	double tmat[9], A[16], B[16], C[16];
	// C = composition of tensor matrix and translation
	TEN_LIST2MAT(tmat, tdata);
	ELL_43M_INSET(A, tmat);
	//	ELL_4M_SET_IDENTITY(A);
	ELL_4M_SET_TRANSLATE(B, x, y, z);
	ELL_4M_MUL(C, B, A);
	ELL_4M_TRANSPOSE_IP(C, tmp);
	//	printf("glyph at (%d,%d,%d) -> (%g,%g,%g) with transform:\n",
	//	       xi, yi, zi, x, y, z);
	//	ell4mPrint_d(stdout, C);
	Transform *tr = new Transform();
	tr->set(C);

#ifdef USE_GLYPH_GROUP
	glyphs.add(new Glyph(new Instance(new InstanceWrapperObject(obj),tr),
		 c[anisoType]));
#else
	all->add(new Glyph(new Instance(new InstanceWrapperObject(obj),tr),
		 c[anisoType]));
#endif
      }
    }
  }
  printf("%s: created %d glyphs!\n", me, numGlyphs);
#ifdef USE_GLYPH_GROUP
  all->add(new GlyphGroup(glyphs, gridcellsize, num_levels));
  printf("%s: created GlyphGroup\n", me);
#endif

  //////////////////////////////////////////////////////
  // all the scene stuff
  Plane groundplane ( Point(-500, 300, 0), Vector(7, -3, 2) );
  //Color cup(0,0,0.3);
  //Color cdown(0.4, 0.2, 0);
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);
  Camera cam(Point(0,0,400), Point(0,0,0),
	     Vector(0,1,0), 60.0);
  Color bgcolor(0,0,0);
  double ambient_scale = 0.3;
  
  Scene* scene=new Scene(all, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);

  Light *scene_light = new Light(Point(500,-300,300), Color(1,1,1), 0);
  scene_light->name_ = "Glyph light";
  scene->add_light(scene_light);
  scene->select_shadow_mode( Hard_Shadows );
  

  
  // clean up hest memory
  airMopOkay(mop);
  return scene;
}

