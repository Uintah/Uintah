/*
  dtiglyph.cc

  Scene file for rendering DTI tensor data, via Gordon's glyph scheme.

  Authors:  James Bigler (bigler@cs.utah.edu)
            Gordon Kindlmann (gk@cs.utah.edu)
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
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
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

#include <nrrd.h>
#include <hest.h>
#include <air.h>
#include <biff.h>
#include <ten.h>

using namespace rtrt;
using namespace std;


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
  if (nrrdLoad(*nrrdP, str)) {
    airMopAdd(mop, nerr = biffGetDone(NRRD), airFree, airMopOnError);
    if (strlen(nerr) > AIR_STRLEN_HUGE - 1)
      nerr[AIR_STRLEN_HUGE - 1] = '\0';
    strcpy(err, nerr);
    airMopError(mop);
    return 1;
  }
  if (!tenValidTensor(*nrrdP, nrrdTypeFloat, AIR_TRUE)) {
    /* why not use the given err[] as a temp buffer */
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

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  airArray *mop;
  hestOpt *opt = NULL;
  int anisoType;
  char *me, *err;
  float glyphScale;
  Nrrd *nin;
  

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
	     
  mop = airMopInit();
  airMopAdd(mop, opt, (airMopper)hestOptFree, airMopAlways);
  me = argv[0];
  if (argc != 4) {
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
  fprintf(stderr, "%s: got dti volume %d x %d x %d\n", me,
	  nin->axis[1].size, nin->axis[2].size, nin->axis[3].size);

  //////////////////////////////////////////////////////
  // add geometry to this :)
  Group *all = new Group();

  //  PhongMaterial *matl = new PhongMaterial(Color(1,0,0), 1);
  Phong *matl = new Phong(Color(1,0,0), Color(1,1,1), 100);
  Sphere *obj = new Sphere(matl, Point(0,0,0), glyphScale);
  Transform *tr = new Transform();
  double t[16];
  ELL_4M_SET_IDENTITY(t);
  tr->set(t);
  all->add(new Instance(new InstanceWrapperObject(obj),tr));

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

  Light *scene_light = new Light(Point(500,-300,300), Color(.8,.8,.8), 0);
  scene_light->name_ = "Glyph light";
  scene->add_light(scene_light);
  scene->select_shadow_mode( Hard_Shadows );
  

  // clean up hest memory
  airMopOkay(mop);
  return scene;
}
