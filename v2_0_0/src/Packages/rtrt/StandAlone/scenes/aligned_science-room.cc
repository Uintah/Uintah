#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Mesh.h>
#include <Packages/rtrt/Core/ASEReader.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/Bezier.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Speckle.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Core/Math/MinMax.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/TexturedTri.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/UVCylinderArc.h>
#include <Packages/rtrt/Core/UVCylinder.h>


#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/HVolumeBrick16.h>
#include <Packages/rtrt/Core/MIPHVB16.h>
#include <Packages/rtrt/Core/CutVolumeDpy.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/ColorMapDpy.h>
#include <Packages/rtrt/Core/PhongColorMapMaterial.h>
#include <Packages/rtrt/Core/CutMaterial.h>
#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#include <Packages/rtrt/Core/DynamicInstance.h>

using namespace rtrt;
using namespace std;
using SCIRun::Thread;

#define ADD_BRICKBRACK
//#define ADD_VIS_FEM
//#define ADD_HEAD
//#define ADD_CSAFE_FIRE
//#define ADD_GEO_DATA
//#define ADD_SHEEP
//#define ADD_DTIGLYPH
//#define ADD_BOX
#define ADD_SPHERE

#ifdef ADD_DTIGLYPH

#include <Packages/rtrt/Core/Glyph.h>
#include <teem/nrrd.h>
#include <teem/hest.h>
#include <teem/air.h>
#include <teem/biff.h>
#include <teem/ten.h>

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

void make_brain_glyphs(Group *g, const Array1<Light *> &pml, int argc, char *argv[100]) {
  airArray *mop;
  hestOpt *opt = NULL;
  int anisoType;
  char *me, *err;
  float glyphScale, anisoThresh;
  Nrrd *nin;
  int gridcellsize;
  int num_levels;

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
  hestOptAdd(&opt, "gridcellsize", "gridcellsize", airTypeInt, 1, 1,
	     &gridcellsize, "3",
	     "size of the grid cells to put around the GlyphGroup");
  hestOptAdd(&opt, "nl", "num_levels", airTypeInt, 1, 1, &num_levels, "10",
	     "number of grid levels to use for optimizations");
  
  mop = airMopInit();
  airMopAdd(mop, opt, (airMopper)hestOptFree, airMopAlways);
  me = argv[0];
  if (argc < 5) {
    hestInfo(stderr, me, dtiINFO, NULL);
    hestUsage(stderr, opt, me, NULL);
    hestGlossary(stderr, opt, NULL);
    airMopError(mop);
    exit(-1);
  }
  fprintf(stderr, "%s: reading input ... ", me); fflush(stderr);
  if (hestParse(opt, argc-1, argv+1, &err, NULL)) {
    fprintf(stderr, "%s: %s\n", me, err); free(err);
    hestUsage(stderr, opt, me, NULL);
    hestGlossary(stderr, opt, NULL);
    airMopError(mop);
    exit(-1);
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

  int sx, sy, sz,        // sizes along x,y,z axes
    xi, yi, zi,          // indices into x,y,z axes
    numGlyphs;
  float zs, ys, xs,      // spacings along x,y,z axes
    as;                  // average spacing
  sx = nin->axis[1].size;
  sy = nin->axis[2].size;
  sz = nin->axis[3].size;
  xs = nin->axis[1].spacing;
  ys = nin->axis[2].spacing;
  zs = nin->axis[3].spacing;
  as = (xs + ys + zs)/3.0;
  fprintf(stderr, "%s: average spacing = %g\n", me, as);
  float tmp,             // don't ask
    *tdata,              // all tensor data; 7 floats per tensor
    x, y, z,             // world-ish position (scaled by spacings)
    eval[3], evec[9],    // eigen{values,vectors} of tensor
    c[TEN_ANISO_MAX+1];  // all possible anisotropies
  tdata = (float*)nin->data;
  numGlyphs = 0;
  Array1<Glyph*> glyphs;
  for (zi = 0; zi < sz; zi++) {
    z = zs * zi;
    for (yi = 0; yi < sy; yi++) {
      y = ys * yi;
      for (xi = 0; xi < sx; xi++, tdata+=7) {
	x = xs * xi;

	// we always ignore data points with low confidence
	if (!( tdata[0] > 0.75 ))
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
	for (int l=0; l<pml.size(); l++)
	  matl->my_lights.add(pml[l]);

	// These are cool transparent/reflective glyphs
	//PhongMaterial *matl = new PhongMaterial(rgb, 0.3, 0.4, 100, true);
	// all glyphs start at the origin

	Sphere *obj = new Sphere(matl, Point(0,0,0), glyphScale);


	double tmat[9], A[16], B[16], C[16];
	// C = composition of tensor matrix and translation

	TEN_LIST2MAT(tmat, tdata);
	ELL_43M_INSET(A, tmat);
	ELL_4M_SET_SCALE(B, as, as, as);
	// C = composition of uniform scaling and tensor matrix
	ELL_4M_MUL(C, B, A);
	ELL_4M_SET_TRANSLATE(A,
			     x - xs*(sx-1)/2,
			     y - ys*(sy-1)/2,
			     zs*(sz-1) - z);
	// B = composition of translation and C
	ELL_4M_MUL(B, A, C);
	ELL_4M_TRANSPOSE_IP(B, tmp);
	//	printf("glyph at (%d,%d,%d) -> (%g,%g,%g) with transform:\n",
	//	       xi, yi, zi, x, y, z);
	//	ell4mPrint_d(stdout, C);
	Transform *tr = new Transform();
	tr->set(B);

	tr->pre_scale(Vector(.015,.015,.015));
	tr->pre_translate(Vector(-8, 8, 0.55));
	glyphs.add(new Glyph(new Instance(new InstanceWrapperObject(obj),tr),
		 c[anisoType]));
      }
    }
  }
  printf("%s: created %d glyphs!\n", me, numGlyphs);
  g->add(new GlyphGroup(glyphs, gridcellsize, num_levels));
  printf("%s: created GlyphGroup\n", me);
}
#endif

void make_walls_and_posters(Group *g, const Point &center, 
			    const Array1<Light *> &room_lights) {
  Vector north(0,1,0);
  Vector east(1,0,0);
  Vector up(0,0,1);
  double east_west_wall_length=8;
  double north_south_wall_length=8;
  double wall_height=4;
  double door_height=2.3;
  double door_width=1.7;
  double door_inset_distance=1.15;
  double wall_thickness=0.2;
  double fc_thickness=0.001;

  Group* north_wall=new Group();
  Group* west_wall=new Group();
  Group* south_wall=new Group();
  Group* east_wall=new Group();
  Group* ceiling_floor=new Group();

  ImageMaterial *stucco = new ImageMaterial("/opt/SCIRun/data/Geometry/textures/science-room/stucco.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  ImageMaterial *seawalls = new ImageMaterial("/opt/SCIRun/data/Geometry/models/livingroom/livingroom-obj/maps/brick.ppm",
		      ImageMaterial::Tile, ImageMaterial::Tile,
		      1, Color(0,0,0), 0);
					      
  int l;
  for (l=0; l<room_lights.size(); l++)
    stucco->my_lights.add(room_lights[l]);

  Point north_wall_center(center+east_west_wall_length/2*north+
			  wall_height/2*up);
  north_wall->add(new Rect(stucco, north_wall_center,
			   4*east, 2*up));
  north_wall->add(new Rect(stucco, north_wall_center+north*wall_thickness,
			   (4+wall_thickness)*east,(2+fc_thickness)*up));
  Point west_wall_center(center-north_south_wall_length/2*east+
			 wall_height/2*up);
  west_wall->add(new Rect(stucco, west_wall_center,
			  4*north, 2*up));
  west_wall->add(new Rect(stucco, west_wall_center-east*wall_thickness,
			  (4+wall_thickness)*north,(2+fc_thickness)*up));

  Point south_wall_center(center-east_west_wall_length/2*north+
			  wall_height/2*up);
  Point south_floor_west_corner(center-east_west_wall_length/2*north-
				north_south_wall_length/2*east);
  Point south_floor_east_corner(south_floor_west_corner+
				north_south_wall_length*east);
  Point south_ceiling_west_corner(south_floor_west_corner+
				  wall_height*up);
  Point south_ceiling_east_corner(south_floor_east_corner+
				  wall_height*up);

  Point south_floor_west_out_corner(south_floor_west_corner-
				    north*wall_thickness-
				    east*wall_thickness-
				    up*fc_thickness);
  Point south_floor_east_out_corner(south_floor_east_corner-
				    north*wall_thickness+
				    east*wall_thickness-
				    up*fc_thickness);
  Point south_ceiling_west_out_corner(south_ceiling_west_corner-
				    north*wall_thickness-
				    east*wall_thickness+
				    up*fc_thickness);
  Point south_ceiling_east_out_corner(south_ceiling_east_corner-
				    north*wall_thickness+
				    east*wall_thickness+
				    up*fc_thickness);
				      
  Point north_floor_west_corner(center+east_west_wall_length/2*north-
				north_south_wall_length/2*east);
  Point north_floor_east_corner(north_floor_west_corner+
				north_south_wall_length*east);
  Point north_ceiling_west_corner(north_floor_west_corner+
				  wall_height*up);
  Point north_ceiling_east_corner(north_floor_east_corner+
				  wall_height*up);

  Point north_floor_west_out_corner(north_floor_west_corner+
				    north*wall_thickness-
				    east*wall_thickness-
				    up*fc_thickness);
  Point north_floor_east_out_corner(north_floor_east_corner+
				    north*wall_thickness+
				    east*wall_thickness-
				    up*fc_thickness);
  Point north_ceiling_west_out_corner(north_ceiling_west_corner+
				    north*wall_thickness-
				    east*wall_thickness+
				    up*fc_thickness);
  Point north_ceiling_east_out_corner(north_ceiling_east_corner+
				    north*wall_thickness+
				    east*wall_thickness+
				    up*fc_thickness);

  Material *gray = new LambertianMaterial(Color(0.3,0.3,0.3));
  for (l=0; l<room_lights.size(); l++)
    gray->my_lights.add(room_lights[l]);

  double cable_radius=0.02;
  Point north_cable_1_base(north_floor_west_corner+
			   east*east_west_wall_length/4-north*.08);
  Point north_cable_2_base(north_cable_1_base+east_west_wall_length/4*east);
  Point north_cable_3_base(north_cable_2_base+east_west_wall_length/4*east);
  north_wall->add(new Cylinder(gray, north_cable_1_base,
			       north_cable_1_base+up*wall_height, 
			       cable_radius));
  north_wall->add(new Cylinder(gray, north_cable_2_base,
			       north_cable_2_base+up*wall_height, 
			       cable_radius));
  north_wall->add(new Cylinder(gray, north_cable_3_base,
			       north_cable_3_base+up*wall_height, 
			       cable_radius));

  Material* dnaM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/DNA2.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  north_wall->add(new Rect(dnaM, north_cable_1_base+up*2.4-
			   north*(cable_radius+.005), east*.93, up*-1.2));
  for (l=0; l<room_lights.size(); l++)
    dnaM->my_lights.add(room_lights[l]);

  Material* hypatiaM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/Hypatia.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  north_wall->add(new Rect(hypatiaM, north_cable_2_base+up*3.0-
			   north*(cable_radius+.005), east*.55, up*-.6));
  for (l=0; l<room_lights.size(); l++)
    hypatiaM->my_lights.add(room_lights[l]);

  Material* emilieM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/emilie_du_chatelet.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  north_wall->add(new Rect(emilieM, north_cable_2_base+up*1.6-
			   north*(cable_radius+.005), east*.45, up*-.55));
  for (l=0; l<room_lights.size(); l++)
    emilieM->my_lights.add(room_lights[l]);

  Material* bluesunM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/bluesun.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  for (l=0; l<room_lights.size(); l++)
    bluesunM->my_lights.add(room_lights[l]);

  north_wall->add(new Rect(bluesunM, north_cable_3_base+up*2.5-
			   north*(cable_radius+.005), east*1.1, up*-1.1));

  Point west_cable_1_base(south_floor_west_corner+
			   north*north_south_wall_length/5+east*.08);
  Point west_cable_2_base(west_cable_1_base+north_south_wall_length/5*north);
  Point west_cable_3_base(west_cable_2_base+north_south_wall_length/5*north);
  Point west_cable_4_base(west_cable_3_base+north_south_wall_length/5*north);
  west_wall->add(new Cylinder(gray, west_cable_1_base,
			      west_cable_1_base+up*wall_height, 
			      cable_radius));
  west_wall->add(new Cylinder(gray, west_cable_2_base,
			      west_cable_2_base+up*wall_height, 
			      cable_radius));
  west_wall->add(new Cylinder(gray, west_cable_3_base,
			      west_cable_3_base+up*wall_height, 
			      cable_radius));
  west_wall->add(new Cylinder(gray, west_cable_4_base,
			      west_cable_4_base+up*wall_height, 
			      cable_radius));

  Material* galileoM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/galileo.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(galileoM, west_cable_1_base+up*2.3+
			  east*(cable_radius+.005), north*.95, up*-1.4));
  for (l=0; l<room_lights.size(); l++)
    galileoM->my_lights.add(room_lights[l]);
  
  Material* brunoM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/bruno.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(brunoM, west_cable_2_base+up*2.95+
			  east*(cable_radius+.005), north*.43, up*-0.55));
  for (l=0; l<room_lights.size(); l++)
    brunoM->my_lights.add(room_lights[l]);

  Material* maxwellM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/james_clerk_maxwell.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(maxwellM, west_cable_2_base+up*1.6+
			  east*(cable_radius+.005), north*.4, up*-0.5));
  for (l=0; l<room_lights.size(); l++)
    maxwellM->my_lights.add(room_lights[l]);

  Material* joeM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/joe_head.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(joeM, west_cable_3_base+up*1.65+
			  east*(cable_radius+.005), north*.7, up*-0.75));
  for (l=0; l<room_lights.size(); l++)
    joeM->my_lights.add(room_lights[l]);

  Material* australM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/australopithecus_boisei.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(australM, west_cable_4_base+up*1.65+
			  east*(cable_radius+.005), north*.62, up*-0.75));
  for (l=0; l<room_lights.size(); l++)
    australM->my_lights.add(room_lights[l]);

  Material* apolloM =
    new ImageMaterial("/opt/SCIRun/data/Geometry/models/science-room/posters2/Apollo16_lander.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(apolloM, west_cable_3_base+up*3.2+north*.75+
			  east*(cable_radius+.005), north*1.5, up*-0.5));
  for (l=0; l<room_lights.size(); l++)
    apolloM->my_lights.add(room_lights[l]);

// rectangle that goes across the whole wall, over the doorway
//
//    1 -------------------- 6 (from outside)
//      |                  |
//      |  7  8            |
//      |  ----            |
//      |  |  |            |
//      |  |  |            |
//      |-------------------
//     2   3  4             5

  Point p1,p2,p3,p4,p5,p6,p7,p8;
  Point p1t,p2t,p3t,p4t,p5t,p6t,p7t,p8t;
  Point p1o,p2o,p3o,p4o,p5o,p6o,p7o,p8o;
  Point p1to,p2to,p3to,p4to,p5to,p6to,p7to,p8to;
  TexturedTri *tri;

  double rx,ry,rz;
  rx=ry=8*2;
  rz=4*2;
  p1=Point(south_ceiling_west_corner);
  p1t=Point(1,1,0);
  p2=Point(south_floor_west_corner);
  p2t=Point(1,0,0);
  p3=Point(south_floor_west_corner+door_inset_distance*east);
  p3t=Point(1-door_inset_distance/north_south_wall_length,0,0);
  p4=Point(p3+door_width*east);
  p4t=Point(1-(door_inset_distance+door_width)/north_south_wall_length,0,0);
  p5=Point(south_floor_east_corner);
  p5t=Point(0,0,0);
  p6=Point(south_ceiling_east_corner);
  p6t=Point(0,1,0);
  p7=Point(p3+door_height*up);
  p7t=Point(p3t.x(),door_height/wall_height,0);
  p8=Point(p4+door_height*up);
  p8t=Point(p4t.x(),door_height/wall_height,0);

  p1to=Point(p1t.x()*rx,p1t.y()*rz,0);
  p2to=Point(p2t.x()*rx,p2t.y()*rz,0);
  p3to=Point(p3t.x()*rx,p3t.y()*rz,0);
  p4to=Point(p4t.x()*rx,p4t.y()*rz,0);
  p5to=Point(p5t.x()*rx,p5t.y()*rz,0);
  p6to=Point(p6t.x()*rx,p6t.y()*rz,0);
  p7to=Point(p7t.x()*rx,p7t.y()*rz,0);
  p8to=Point(p8t.x()*rx,p8t.y()*rz,0);

  p1o=Point(p1-north*wall_thickness-east*wall_thickness+up*fc_thickness);
  p2o=Point(p2-north*wall_thickness-east*wall_thickness-up*fc_thickness);
  p3o=Point(p3-north*wall_thickness-up*fc_thickness);
  p4o=Point(p4-north*wall_thickness-up*fc_thickness);
  p5o=Point(p5-north*wall_thickness+east*wall_thickness-up*fc_thickness);
  p6o=Point(p6-north*wall_thickness+east*wall_thickness+up*fc_thickness);
  p7o=Point(p7-north*wall_thickness);
  p8o=Point(p8-north*wall_thickness);

  Point s3(p3),s4(p4),s7(p7),s8(p8);
  Point s3o(p3o), s4o(p4o), s7o(p7o), s8o(p8o);

  tri = new TexturedTri(stucco, p1, p2, p3);
  tri->set_texcoords(p1t,p2t,p3t);
  south_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p2o, p3o);
  tri->set_texcoords(p1to,p2to,p3to);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p3, p7);
  tri->set_texcoords(p1t,p3t,p7t);
  south_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p3o, p7o);
  tri->set_texcoords(p1to,p3to,p7to);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p7, p8);
  tri->set_texcoords(p1t,p7t,p8t);
  south_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p7o, p8o);
  tri->set_texcoords(p1to,p7to,p8to);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p8, p6);
  tri->set_texcoords(p1t,p8t,p6t);
  south_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p8o, p6o);
  tri->set_texcoords(p1to,p8to,p6to);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p5, p8, p6);
  tri->set_texcoords(p5t,p8t,p6t);
  south_wall->add(tri);
  tri = new TexturedTri(seawalls, p5o, p8o, p6o);
  tri->set_texcoords(p5to,p8to,p6to);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p4, p8, p5);
  tri->set_texcoords(p4t,p8t,p5t);
  south_wall->add(tri);
  tri = new TexturedTri(seawalls, p4o, p8o, p5o);
  tri->set_texcoords(p4to,p8to,p5to);
  south_wall->add(tri);

  p1=Point(north_ceiling_east_corner);
  p1t=Point(0,1,0);
  p2=Point(north_floor_east_corner);
  p2t=Point(0,0,0);
  p3=Point(north_floor_east_corner-door_inset_distance*north);
  p3t=Point(door_inset_distance/east_west_wall_length,0,0);
  p4=Point(p3-door_width*north);
  p4t=Point((door_inset_distance+door_width)/east_west_wall_length,0,0);
  p5=Point(south_floor_east_corner);
  p5t=Point(1,0,0);
  p6=Point(south_ceiling_east_corner);
  p6t=Point(1,1,0);
  p7=Point(p3+door_height*up);
  p7t=Point(p3t.x(),door_height/wall_height,0);
  p8=Point(p4+door_height*up);
  p8t=Point(p4t.x(),door_height/wall_height,0);

  p1to=Point(p1t.x()*ry,p1t.y()*rz,0);
  p2to=Point(p2t.x()*ry,p2t.y()*rz,0);
  p3to=Point(p3t.x()*ry,p3t.y()*rz,0);
  p4to=Point(p4t.x()*ry,p4t.y()*rz,0);
  p5to=Point(p5t.x()*ry,p5t.y()*rz,0);
  p6to=Point(p6t.x()*ry,p6t.y()*rz,0);
  p7to=Point(p7t.x()*ry,p7t.y()*rz,0);
  p8to=Point(p8t.x()*ry,p8t.y()*rz,0);

  p1o=Point(p1+north*wall_thickness+east*wall_thickness+up*fc_thickness);
  p2o=Point(p2+north*wall_thickness+east*wall_thickness-up*fc_thickness);
  p3o=Point(p3+east*wall_thickness-up*fc_thickness);
  p4o=Point(p4+east*wall_thickness-up*fc_thickness);
  p5o=Point(p5-north*wall_thickness+east*wall_thickness-up*fc_thickness);
  p6o=Point(p6-north*wall_thickness+east*wall_thickness+up*fc_thickness);
  p7o=Point(p7+east*wall_thickness);
  p8o=Point(p8+east*wall_thickness);

  Point e3(p3), e4(p4), e7(p7), e8(p8); 
  Point e3o(p3o), e4o(p4o), e7o(p7o), e8o(p8o);
  
  tri = new TexturedTri(stucco, p1, p2, p3);
  tri->set_texcoords(p1t,p2t,p3t);
  east_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p2o, p3o);
  tri->set_texcoords(p1to,p2to,p3to);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p3, p7);
  tri->set_texcoords(p1t,p3t,p7t);
  east_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p3o, p7o);
  tri->set_texcoords(p1to,p3to,p7to);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p7, p8);
  tri->set_texcoords(p1t,p7t,p8t);
  east_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p7o, p8o);
  tri->set_texcoords(p1to,p7to,p8to);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p8, p6);
  tri->set_texcoords(p1t,p8t,p6t);
  east_wall->add(tri);
  tri = new TexturedTri(seawalls, p1o, p8o, p6o);
  tri->set_texcoords(p1to,p8to,p6to);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p5, p8, p6);
  tri->set_texcoords(p5t,p8t,p6t);
  east_wall->add(tri);
  tri = new TexturedTri(seawalls, p5o, p8o, p6o);
  tri->set_texcoords(p5to,p8to,p6to);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p4, p8, p5);
  tri->set_texcoords(p4t,p8t,p5t);
  east_wall->add(tri);
  tri = new TexturedTri(seawalls, p4o, p8o, p5o);
  tri->set_texcoords(p4to,p8to,p5to);
  east_wall->add(tri);

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));
  for (l=0; l<room_lights.size(); l++)
    white->my_lights.add(room_lights[l]);
  Material *bb_matl = new Phong(Color(0.45,0.45,0.45), Color(0.3,0.3,0.3), 20, 0);
  for (l=0; l<room_lights.size(); l++)
    bb_matl->my_lights.add(room_lights[l]);

  UVCylinderArc *uvc;

  uvc=new UVCylinderArc(bb_matl, north_floor_west_corner+north*.05,
			north_floor_east_corner+north*.05, 0.1);
  uvc->set_arc(0,M_PI/2);
  north_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_floor_west_corner-east*.05,
			south_floor_west_corner-east*.05, 0.1);
  uvc->set_arc(M_PI,3*M_PI/2);
  west_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_ceiling_west_corner+north*.05,
			north_ceiling_east_corner+north*.05, 0.1);
  uvc->set_arc(M_PI/2,M_PI);
  north_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_ceiling_west_corner-east*.05,
			south_ceiling_west_corner-east*.05, 0.1);
  uvc->set_arc(M_PI/2, M_PI);
  west_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, south_ceiling_west_corner-north*.05,
			south_ceiling_east_corner-north*.05, 0.1);
  uvc->set_arc(M_PI,3*M_PI/2);
  south_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_ceiling_east_corner+east*.05,
			south_ceiling_east_corner+east*.05, 0.1);
  uvc->set_arc(0,M_PI/2);
  east_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, south_floor_west_corner-north*.05,
			s3-north*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);
  south_wall->add(uvc);
  uvc=new UVCylinderArc(bb_matl, s4-north*.05, 
			south_floor_east_corner-north*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);
  south_wall->add(uvc);

  Material *black = new Phong(Color(0.1,0.1,0.1), Color(0.3,0.3,0.3), 20, 0);
  for (l=0; l<room_lights.size(); l++)
    black->my_lights.add(room_lights[l]);

  south_wall->add(new Box(black,s3-north*wall_thickness-north*.1-east*.1,
			  s7+north*.1+east*.1+up*.1));
  south_wall->add(new Box(black,s7-north*wall_thickness-north*.1-east*.1-up*.1,
			  s8+north*.1+east*.1+up*.1));
  south_wall->add(new Box(black,s4-north*wall_thickness-north*.1-east*.1,
			  s8+north*.1+east*.1+up*.1));
  
  uvc=new UVCylinderArc(bb_matl, north_floor_east_corner+east*.05,
			e3+east*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);
  east_wall->add(uvc);
  uvc=new UVCylinderArc(bb_matl, e4+east*.05, 
			south_floor_east_corner+east*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);  
  east_wall->add(uvc);

  east_wall->add(new Box(black,e3-north*.1-east*.1, 
			 e7+north*.1+east*.1+up*.1+east*wall_thickness));
  east_wall->add(new Box(black,e8-north*.1-east*.1-up*.1,
			 e7+north*.1+east*.1+up*.1+east*wall_thickness));
  east_wall->add(new Box(black,e4-north*.1-east*.1,
			 e8+north*.1+east*.1+up*.1+east*wall_thickness));

  east_wall->add(new UVCylinderArc(bb_matl, e3+east*.05, e7+east*.05, 0.1));
  east_wall->add(new UVCylinderArc(bb_matl, e7+east*.05, e8+east*.05, 0.1));
  east_wall->add(new UVCylinderArc(bb_matl, e8+east*.05, e4+east*.05, 0.1));
  
  // add the ceiling
  ceiling_floor->add(new Rect(white, Point(-8, 8, 4),
		       Vector(4, 0, 0), Vector(0, 4, 0)));
  ceiling_floor->add(new Rect(white, Point(-8, 8, 4)+up*fc_thickness,
			      Vector(4+wall_thickness, 0, 0), 
			      Vector(0, 4+wall_thickness, 0)));

  // table top
  ImageMaterial *cement_floor = 
    new ImageMaterial("/opt/SCIRun/data/Geometry/textures/science-room/cement-floor.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp, 0,
		      Color(0,0,0), 0);
  for (l=0; l<room_lights.size(); l++)
    cement_floor->my_lights.add(room_lights[l]);
  Object* floor=new Rect(cement_floor, Point(-8, 8, 0),
			 Vector(4, 0, 0), Vector(0, 4, 0));
  ceiling_floor->add(floor);

  g->add(ceiling_floor);
  g->add(north_wall);
  g->add(west_wall);
  g->add(south_wall);
  g->add(east_wall);
}

SpinningInstance *make_dna(Group *g, Group *s, 
			   const Array1<Light *> &room_lights) { 
  int l;
  Phong *red = new Phong(Color(0.8,0.2,0.2), Color(0.5,0.5,0.5), 40, 0.4);
  for (l=0; l<room_lights.size(); l++)
    red->my_lights.add(room_lights[l]);
  Phong *yellow = new Phong(Color(0.8,0.8,0.2), Color(0.5,0.5,0.5), 40, 0.4);
  for (l=0; l<room_lights.size(); l++)
    yellow->my_lights.add(room_lights[l]);
  Phong *green = new Phong(Color(0.2,0.8,0.2), Color(0.5,0.5,0.5), 40, 0.4);
  for (l=0; l<room_lights.size(); l++)
    green->my_lights.add(room_lights[l]);
  Phong *blue = new Phong(Color(0.2,0.2,0.8), Color(0.5,0.5,0.5), 40, 0.4);
  for (l=0; l<room_lights.size(); l++)
    blue->my_lights.add(room_lights[l]);
  Phong *white = new Phong(Color(0.3,0.3,0.3), Color(0.3,0.3,0.3), 20, 0);
  for (l=0; l<room_lights.size(); l++)
    white->my_lights.add(room_lights[l]);
  Point center_base(-8+3.7,8+3.7,0.63);
  Vector up(0,0,1);
  Vector left(1,0,0);
  Vector in(0,1,0);
  double rad=0.1;
  double dh=0.04;
  double th0=M_PI/4;
  double dth=M_PI/6;
  double srad=0.025;
  double crad=0.009;

  Group *g1 = new Group;
  for (int i=0; i<14; i++) {
    Phong *l, *r;
//    if (drand48() < 0.5) { l=red; r=green; }
//    else { l=yellow; r=blue; }
    double rand=drand48();
    if (rand < 0.25) { l=red; r=green; }
    else if (rand < 0.5) { l=green; r=red; }
    else if (rand < 0.75) { l=yellow; r=blue; }
    else { l=blue; r=yellow; }
    Vector v(left*(rad*cos(th0+dth*i))+in*(rad*sin(th0+dth*i)));
    Point ls(center_base+up*(dh*i)+v);
    Point rs(center_base+up*(dh*i)-v);
    g1->add(new Sphere(l, ls, srad));
    g1->add(new Sphere(r, rs, srad));
    g1->add(new Cylinder(white, ls, rs, crad));
  }

  Grid *g2 = new Grid(g1, 5);
  Transform *mt = new Transform;
  InstanceWrapperObject *mw = new InstanceWrapperObject(g2);
  SpinningInstance *smw = new SpinningInstance(mw, mt, center_base, up, 0.2);
  g->add(smw);
  s->add(smw);
  return smw;
}

void add_objects(Group *g, Group *s, const Point &center, 
		 const Array1<Light *> &room_lights) {
  Transform room_trans;
  room_trans.pre_translate(center.vector());

  string pathname("/opt/SCIRun/data/Geometry/models/science-room/");

  Array1<int> sizes;
  Array1<string> names;

  names.add(string("386dx"));
  sizes.add(32);
  names.add(string("3d-glasses-01"));
  sizes.add(16);
  names.add(string("3d-glasses-02"));
  sizes.add(16);
  names.add(string("abacus"));
  sizes.add(32);
  names.add(string("coffee-cup-01"));
  sizes.add(16);
  names.add(string("coffee-cup-02"));
  sizes.add(16);
  names.add(string("coffee-cup-03"));
  sizes.add(16);
  names.add(string("coffee-cup-04"));
  sizes.add(16);
  names.add(string("coffee-cup-05"));
  sizes.add(16);
  names.add(string("coffee-cup-06"));
  sizes.add(16);
  names.add(string("coffee-cup-07"));
  sizes.add(16);
  names.add(string("coffee-cup-08"));
  sizes.add(16);
  names.add(string("corbusier-01"));
  sizes.add(16);
  names.add(string("corbusier-02"));
  sizes.add(16);
  names.add(string("corbusier-03"));
  sizes.add(16);
  names.add(string("corbusier-04"));
  sizes.add(16);
  names.add(string("corbusier-05"));
  sizes.add(16);
  names.add(string("end-table-01"));
  sizes.add(16);
  names.add(string("end-table-02"));
  sizes.add(16);
  names.add(string("end-table-03"));
  sizes.add(16);
  names.add(string("end-table-04"));
  sizes.add(16);
  names.add(string("end-table-05"));
  sizes.add(16);
  names.add(string("faucet-01"));
  sizes.add(32);
  names.add(string("sink-01"));
  sizes.add(16);
  names.add(string("futuristic-curio1"));
  sizes.add(16);
  names.add(string("futuristic-curio2"));
  sizes.add(16);
  names.add(string("hmd"));
  sizes.add(16);
  names.add(string("microscope"));
  sizes.add(16);
  names.add(string("plant-01"));
  sizes.add(32);

  Array1<Material *> matls;
  int i;
  for (i=0; i<names.size(); i++) {
    cerr << "Reading: "<<names[i]<<"\n";
    string objname(pathname+names[i]+string(".obj"));
    string mtlname(pathname+names[i]+string(".mtl"));
    Group *g1 = new Group;
    if (!readObjFile(objname, mtlname, room_trans, matls, g1, sizes[i]))
      exit(0);
    g->add(g1);
    s->add(g1);
    int m, l;
    for (m=0; m<matls.size(); m++)
      for (l=0; l<room_lights.size(); l++)
	matls[m]->my_lights.add(room_lights[l]);
  }
}

extern "C"
Scene* make_scene(int /*argc*/, char* /*argv*/[], int nworkers)
{
//  for(int i=1;i<argc;i++) {
//    cerr << "Unknown option: " << argv[i] << '\n';
//    cerr << "Valid options for scene: " << argv[0] << '\n';
//    return 0;
//  }

// Start inside:
//  Point Eye(-11, 8, 1.6);
//  Point Lookat(-8, 8, 2.0);
//  Vector Up(0,0,1);
//  double fov=60;

// Start outside:
  //  Point Eye(-10.9055, -0.629515, 1.56536);
  // Point Lookat(-8.07587, 15.7687, 1.56536);
  //Vector Up(0, 0, 1);
  //double fov=60;

// Just table:
//  Point Eye(-7.64928, 6.97951, 1.00543);
//  Point Lookat(-19.9299, 16.5929, -2.58537);
//  Vector Up(0, 0, 1);
//  double fov=35;

// Just vis
  Point Eye(-11.85, 8.05916, 1.30671);
  Point Lookat(-8.83055, 8.24346, 1.21209);
  Vector Up(0,0,1);
  double fov=45;
  Camera cam(Eye,Lookat,Up,fov);

  Point center(-8, 8, 0);
  Group *g=new Group;
  Group *s=new Group;
  int l;

  //PER MATERIAL LIGHTS FOR THE HOLOGRAMS
  Array1<Light *> holo_lights;
//  Light *holo_light1 = new Light(Point(-8, 10, 0.2), Color(0.5,0.3,0.3),0,1,true);
//  Light *holo_light2 = new Light(Point(-9.41, 6.58, 2.2),Color(0.3,0.5,0.3),0,1,true);
//  Light *holo_light3 = new Light(Point(-6.58, 6.58, 3.2),Color(0.3,0.3,0.5),0,1,true);
  Light *holo_light1 = new Light(Point(-8, 11.5, 3.3), Color(0.9,0.9,0.9),0,1,true);
  Light *holo_light2 = new Light(Point(-4.5, 6, 3.3), Color(0.9,0.9,0.9),0,1,true);
  Light *holo_light3 = new Light(Point(-11.5, 6, 3.3),Color(0.9,0.9,0.9),0,1,true);
  holo_light1->name_ = "hololight1";
  holo_light2->name_ = "hololight2";
  holo_light3->name_ = "hololight3";
  holo_lights.add(holo_light1);
  holo_lights.add(holo_light2);
  holo_lights.add(holo_light3);

  Array1<Light *> room_lights;
  Light *science_room_light0 = new Light(Point(-8, 8, 3.9), Color(.5,.5,.5), 0, .3);
  science_room_light0->name_ = "science room overhead";
  room_lights.add(science_room_light0);
  Light *science_room_light1 = new Light(Point(-5, 8, 3), Color(.5,.5,.5), 0, .3);
  science_room_light1->name_ = "science room corner1";
  room_lights.add(science_room_light1);
//  Light *science_room_light1 = new Light(Point(-5, 11, 3), Color(.5,.5,.5), 0);
//  science_room_light1->name_ = "science room corner";
  Light *science_room_light2 = new Light(Point(-8, 5, 3), Color(.5,.5,.5), 0, .3);
  science_room_light2->name_ = "science room corner2";
  room_lights.add(science_room_light2);

#ifdef ADD_DTIGLYPH
  Group *glyphs = new Group;
  int ac=9;
  char *av[100]= 
    {"rtrt",
     "/opt/SCIRun/data/Geometry/volumes2/gk/gk2-rcc-b06-mask-s070.nhdr",
     "fa",
     "0.45",
     "0.71",
     "-gridcellsize",
     "100",
     "-nl",
     "2" };
  
  make_brain_glyphs(glyphs, holo_lights, ac, av);
#endif

#ifdef ADD_BRICKBRACK
  make_walls_and_posters(g, center, room_lights);
#endif

  Group* table=new Group();

  ImageMaterial *cement_pedestal = 
    new ImageMaterial("/opt/SCIRun/data/Geometry/textures/science-room/cement-pedestal.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp, 0,
		      Color(0,0,0), 0);
  
  table->add(new UVCylinder(cement_pedestal, center,
			    center+Vector(0,0,0.5), 1.5));
  Material *silver = new MetalMaterial(Color(0.25,0.25,0.25), 12);
  table->add(new Disc(silver, center+Vector(0,0,0.5),
		      Vector(0,0,1), 1.5));
  g->add(table);
  s->add(table);

#ifdef ADD_BOX
  Group *boxg = new Group;

  Material *gray = new LambertianMaterial(Color(0.3,0.3,0.3));
  boxg->add(new Box(gray, center+Vector(0,0,2)-Vector(1,1,1),
		    center+Vector(0,0,2)+Vector(1,1,1)));
#endif

//#ifdef ADD_SPHERE
  Group *sphereg = new Group;

  Material *gray2 = new LambertianMaterial(Color(0.3,0.3,0.3));
  sphereg->add(new Sphere(gray2, center+Vector(0,0,2), 0.8));
//#endif

#ifdef ADD_BRICKBRACK
  add_objects(g, s, center, room_lights);
#endif

#ifdef ADD_BRICKBRACK
  SpinningInstance *smw = make_dna(g, s, room_lights);
#endif



  //ADD THE VISIBLE FEMALE DATASET
#ifdef ADD_VIS_FEM
  CutPlaneDpy* vcpdpy=new CutPlaneDpy(Vector(.707,-.707,0), Point(-8,8,1.56));

  ColorMap *vcmap = new ColorMap("/opt/SCIRun/data/Geometry/volumes2/vfem",256);
  Material* vmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  for (l=0; l<holo_lights.size(); l++)
    vmat->my_lights.add(holo_lights[l]);

  Material *vcutmat = new CutMaterial(vmat, vcmap, vcpdpy);
  for (l=0; l<holo_lights.size(); l++)
    vcutmat->my_lights.add(holo_lights[l]);

  CutVolumeDpy* vcvdpy = new CutVolumeDpy(1200.5, vcmap);
  
  HVolumeBrick16* slc0=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/opt/SCIRun/data/Geometry/volumes2/vfem16_0",
					  3, nworkers);
  
  HVolumeBrick16* slc1=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/opt/SCIRun/data/Geometry/volumes2/vfem16_1",
					  3, nworkers);
  
  HVolumeBrick16* slc2=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/opt/SCIRun/data/Geometry/volumes2/vfem16_2",
					  3, nworkers);

  HVolumeBrick16* slc3=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/opt/SCIRun/data/Geometry/volumes2/vfem16_3",
					  3, nworkers);
  
  HVolumeBrick16* slc4=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/opt/SCIRun/data/Geometry/volumes2/vfem16_4",
					  3, nworkers);
  
  HVolumeBrick16* slc5=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/opt/SCIRun/data/Geometry/volumes2/vfem16_5",
					  3, nworkers);

  HVolumeBrick16* slc6=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/opt/SCIRun/data/Geometry/volumes2/vfem16_6",
					  3, nworkers);
					  

  //5cm floating above pedasta;
#define HFLT 0.05
  //allow to scale from pedestal top to 3m up
#define EXTZ 3-(0.5+HFLT)
  //max x,y that will rotate within a 1.5rad circle 2*sqrt(1.5^2/2)=2.22
#define EXTX 2.22
#define EXTY 2.22

  //now where is center of holospace?
#define CENX -8
#define CENY 8
#define CENZ EXTZ/2.0 + 0.5+HFLT

  Group *vig = new Group();
  vig->add(slc0);
  vig->add(slc1);
  vig->add(slc2);
  vig->add(slc3);
  vig->add(slc4);
  vig->add(slc5);
  vig->add(slc6);
  InstanceWrapperObject *viw = new InstanceWrapperObject(vig);

  Transform *vtrans = new Transform();
  vtrans->pre_rotate(3.14/2.0, Vector(0,1,0));
  //after rotation...
#define ORIGEX .48
#define ORIGEY .48
#define ORIGEZ 1.7330
  //choose maximum from model, in this case z
#define SCALE EXTZ/ORIGEZ
  vtrans->pre_scale(Vector(SCALE,SCALE,SCALE));
  vtrans->pre_translate(Vector(CENX, CENY, CENZ));
  //#define ZMIN CENZ-(ORIGZ*SCALE)/2.0
  //#define LOWERZ 0.5+HFLT - ZMIN
  //no need, we've scaled in this case to fit z
#define LOWERZ 0
  vtrans->pre_translate(Vector(0,0,LOWERZ)); //place at HFLT above surface
#undef ORIGEX
#undef ORIGEY
#undef ORIGEZ
#undef SCALE
#undef ZMIN
#undef LOWERZ

  SpinningInstance *vinst = new SpinningInstance(viw, vtrans, Point(-8,8,1.56), Vector(0,0,1), 0.1);
  vinst->set_name("Spinning Visible Woman");

  CutGroup *vcut = new CutGroup(vcpdpy, true);
  vcut->add(vinst);
  vcut->set_name("Visible Femarle Cut Plane");

  vinst->addCPDpy(vcpdpy);
#endif

#ifdef ADD_HEAD
  //ADD THE HEAD DATA SET
  CutPlaneDpy* hcpdpy=new CutPlaneDpy(Vector(.707,-.707,0), Point(-8,8,1.56));

  ColorMap *hcmap = new ColorMap("/opt/SCIRun/data/Geometry/volumes2/head",256);
  Material *hmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  for (l=0; l<holo_lights.size(); l++)
    hmat->my_lights.add(holo_lights[l]);

  Material *hcutmat = new CutMaterial(hmat, hcmap, hcpdpy);
  for (l=0; l<holo_lights.size(); l++)
    hcutmat->my_lights.add(holo_lights[l]);

  CutVolumeDpy* hcvdpy = new CutVolumeDpy(11000.0, hcmap);

  HVolumeBrick16* head=new HVolumeBrick16(hcutmat, hcvdpy,
					  //    "/opt/SCIRun/data/Geometry/volumes2/dave",
					  "/opt/SCIRun/data/Geometry/volumes2/gk2-anat-US.raw",
					      3, nworkers);
  InstanceWrapperObject *hiw = new InstanceWrapperObject(head);

  Transform *htrans = new Transform();
  htrans->rotate(Vector(1,0,0), Vector(0,0,-1));
  //after rotation...
#define ORIGEX 1.48
#define ORIGEY 1.9
#define ORIGEZ 1.6
  //max in this case is y
#define SCALE EXTY/ORIGEY
  htrans->pre_scale(Vector(SCALE,SCALE,SCALE));
  htrans->pre_translate(Vector(CENX, CENY, CENZ));
#define ZMIN CENZ-(ORIGZ*SCALE)/2.0
#define LOWERZ 0.5+HFLT - ZMIN
  htrans->pre_translate(Vector(0,0,LOWERZ)); //place at HFLT above surface
#undef ORIGEX
#undef ORIGEY
#undef ORIGEZ
#undef SCALE
#undef ZMIN
#undef LOWERZ


  SpinningInstance *hinst = new SpinningInstance(hiw, htrans, Point(-8,8,1.56), Vector(0,0,1), 0.1);
  
  hinst->set_name("Spinning Brain");

  CutGroup *hcut = new CutGroup(hcpdpy, true);
  hcut->add(hinst);
  hcut->set_name("Brain Cutting Plane");

  hinst->addCPDpy(hcpdpy);
#endif

#ifdef ADD_CSAFE_FIRE
  //ADD THE CSAFE HEPTAINE POOL FIRE DATA SET
  // Create a default color map.
  int color_size = 5;
  Array1<ColorPos> colors(color_size);
  for(int i = 0; i < color_size; i++) {
    colors[i].x = (float)i/(color_size-1);
    colors[i].val = 1;
  }
  colors[0].c = Color(1,0,0);
  colors[1].c = Color(1,1,0);
  colors[2].c = Color(0,1,0);
  colors[3].c = Color(0,1,1);
  colors[4].c = Color(0,0,1);

  ColorMapDpy *fcdpy = new ColorMapDpy(colors);
  //  dpys.add(fcdpy);
  Material *temp_matl = new Phong(Color(1, 0.7, 0.8), Color(1,1,1), 100);
  VolumeDpy *temp_dpy = new VolumeDpy(1000);
  // Don't add the display unless you plan on rendering the temperature field
  //  dpys.add(temp_dpy);
  VolumeDpy *vel_dpy = new VolumeDpy(5.6);
  //  dpys.add(vel_dpy);

  ////////////////////////////////////////////////////////////////
  //  Load up a group
  
  int fstart = 32;
  int fend = 152;
  //int fstart = 64;
  //  int fend = 80;
  int finc = 8; // never less than 8, must be a multiple of 8
  //  int finc = 16; // 0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160
  //  int finc = 24; // 0, 24, 48, 72, 96, 120, 144, 168
  //    int finc = 32; // 0, 32, 64, 96, 128, 160
  //  int finc = 40; // 0, 40, 80, 120, 160
  SelectableGroup *fire_geom = new SelectableGroup(0.5);
  fire_geom->set_name("CSAFE Fire Velocity Magnitude Time Step Selector");
  SelectableGroup *temp_field = new SelectableGroup(0.5);
  temp_field->set_name("CSAFE Fire Temperture Field Time Step Selector");
  // Now since we won't be able to compute the bounding box for stuff
  // that hasn't been loaded yet, we must manually feed it in.
  BBox f_bb(Point(-1,-1,-1), Point(1,1,1));
  InstanceWrapperObject *temp_iw = new InstanceWrapperObject(temp_field, f_bb);
  InstanceWrapperObject *fire_iw = new InstanceWrapperObject(fire_geom, f_bb);
  Transform *fire_trans = new Transform();
  //after rotation...
  fire_trans->rotate(Vector(1,0,0), Vector(0,0,1));
#define ORIGEX 2
#define ORIGEY 2
#define ORIGEZ 2
  //max in this case is y
#define SCALE EXTY/ORIGEY
  htrans->pre_scale(Vector(SCALE,SCALE,SCALE));
  htrans->pre_translate(Vector(CENX, CENY, CENZ));
#define ZMIN CENZ-(ORIGZ*SCALE)/2.0
#define LOWERZ 0.5+HFLT - ZMIN
  htrans->pre_translate(Vector(0,0,LOWERZ)); //place at HFLT above surface
#undef ORIGEX
#undef ORIGEY
#undef ORIGEZ
#undef SCALE
#undef ZMIN
#undef LOWERZ
  /*
  fire_trans->pre_scale(Vector(1.245,1.245,1.245));
  fire_trans->rotate(Vector(1,0,0), Vector(0,0,1));
  fire_trans->pre_translate(Vector(-8, 8, 1.75));
  fire_trans->pre_translate(Vector(0,0,-.00305));
  */

  SpinningInstance *fire_inst = new SpinningInstance(fire_iw, fire_trans, Point(-8,8,1.75), Vector(0,0,1), 0.1);
  fire_inst->set_name("Spinning CSAFE Velocity Magnitude Field");
  SpinningInstance *temp_inst = new SpinningInstance(temp_iw, fire_trans, Point(-8,8,1.75), Vector(0,0,1), 0.1);
  temp_inst->set_name("Spinning CSAFE Temperature Field");

  //  TimeObj *fire_time = new TimeObj(5);
  for(int f = fstart; f <= fend; f+= finc) {
    char buf[1000];
    // load in the temperature field to color by
    sprintf(buf, "/opt/SCIRun/data/Geometry/volumes2/CSAFE/h300_%04df.raw", f);
    //    cout << "Reading "<<buf<<endl;
    // Send 1 for depth since we don't need to render it and preprocess
    // goes much quickley.
    HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > *tempt = 
      new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >
      (temp_matl, temp_dpy, buf, 1, nworkers);
    temp_field->add(tempt);
    
    // Get min and max and attach to ColorMapDpy.
    float min, max;
    tempt->get_minmax(min, max);
    cerr << "min = "<<min<<", max = "<<max<<endl;
    fcdpy->attach(min,max);

    // Create the Material.  We need to put a pointer to the SpinningInstance
    // object, so that the fields will synchronize.
    Material *temp_color_map = new PhongColorMapMaterial
      (temp_inst,
       fcdpy->get_color_transfer_pointer(),
       fcdpy->get_alpha_transfer_pointer());
    // add the permaterial lights
    for (l=0; l<holo_lights.size(); l++)
      temp_color_map->my_lights.add(holo_lights[l]);

    // Load the velocity field that will be used to isosurface.
    sprintf(buf, "/opt/SCIRun/data/Geometry/volumes2/CSAFE/heptane300_velmag_%04d.raw", f);
    HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > *vel =
      new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > >
      (temp_color_map, vel_dpy, buf, 3, nworkers);


    fire_geom->add(vel);
  }

#if 0
  // Can't do cutting plane on time dependant data just yet.  There's a
  // CycleMaterial that may be able to work, but I haven't checked it out.
  CutPlaneDpy* pd=new CutPlaneDpy(Vector(0,1,0), Point(0,0,0));
  dpys.add(pd);
  //  (new Thread(pd, "Cutting plane display thread"))->detach();
  Object *obj = new CutPlane(fire_geom, pd);
  obj->set_matl(hmat);
#endif

#endif // ifdef ADD_CSAFE_FIRE
  
#ifdef ADD_GEO_DATA
  //ADD THE GEOLOGY DATA SET
  CutPlaneDpy* gcpdpy=new CutPlaneDpy(Vector(.707,-.707,0), Point(-8,8,1.56));

  ColorMap *gcmap = new ColorMap("/opt/SCIRun/data/Geometry/volumes2/Seismic/geo",256);
  Material* gmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  for (l=0; l<holo_lights.size(); l++)
    gmat->my_lights.add(holo_lights[l]);

  Material *gcutmat = new CutMaterial(gmat, gcmap, gcpdpy);
  for (l=0; l<holo_lights.size(); l++)
    gcutmat->my_lights.add(holo_lights[l]);

  CutVolumeDpy* gcvdpy = new CutVolumeDpy(16137.7, gcmap);

  HVolumeBrick16* geology=new HVolumeBrick16(gcutmat, gcvdpy,
					     //unfiltered, full data set
					     //"/opt/SCIRun/data/Geometry/volumes2/Seismic/stack-16full.raw",
					     //filtered, cropped data set
					     "/opt/SCIRun/data/Geometry/volumes2/Seismic/stack-chunks01-m2-16.raw",
					      3, nworkers);
  InstanceWrapperObject *giw = new InstanceWrapperObject(geology);

  Transform *gtrans = new Transform();
  gtrans->rotate(Vector(1,0,0), Vector(0,0,-1));
  //after rotation...
#define ORIGEX .46
#define ORIGEY .4
#define ORIGEZ 2
  //choose maximum from model, in this case z
#define SCALE EXTZ/ORIGEZ
  gtrans->pre_scale(Vector(SCALE,SCALE,SCALE));
  gtrans->pre_translate(Vector(CENX, CENY, CENZ));
  //#define ZMIN CENZ-(ORIGZ*SCALE)/2.0
  //#define LOWERZ 0.5+HFLT - ZMIN
  //no need, we've scaled in this case to fit z
#define LOWERZ 0
  gtrans->pre_translate(Vector(0,0,LOWERZ)); //place at HFLT above surface
#undef ORIGEX
#undef ORIGEY
#undef ORIGEZ
#undef SCALE
#undef ZMIN
#undef LOWERZ

  /*
//  gtrans->pre_scale(Vector(1.245,1.245,1.245)); //fit between z=0.51 and 3
  gtrans->pre_scale(Vector(3.735,3.735,0.6225)); //fit between z=0.51 and 1.75
  gtrans->pre_translate(Vector(-8, 8, 1.25));
//  gtrans->pre_translate(Vector(-8, 8, 1.75));
  gtrans->pre_translate(Vector(0,0,0.1)); //place 4 cm above table
  */

  SpinningInstance *ginst = new SpinningInstance(giw, gtrans, Point(-8,8,1.56), Vector(0,0,1), 0.1);
  ginst->set_name("Spinning Geology");

  CutGroup *gcut = new CutGroup(gcpdpy, true);
  gcut->set_name("Geology Cutting Plane");
  gcut->add(ginst);

  ginst->addCPDpy(gcpdpy);
#endif

#ifdef ADD_SHEEP
  //ADD THE SHEEP HEART DATA SET
  CutPlaneDpy* scpdpy=new CutPlaneDpy(Vector(.707,-.707,0), Point(-8,8,1.56));

  ColorMap *scmap = new ColorMap("/opt/SCIRun/data/Geometry/volumes2/sheep",256);
  Material *smat=new LambertianMaterial(Color(0.7,0.7,0.7));
  for (l=0; l<holo_lights.size(); l++)
    smat->my_lights.add(holo_lights[l]);

  Material *scutmat = new CutMaterial(smat, scmap, scpdpy);
  for (l=0; l<holo_lights.size(); l++)
    scutmat->my_lights.add(holo_lights[l]);

  CutVolumeDpy* scvdpy = new CutVolumeDpy(11000.0, scmap);

  HVolumeBrick16* sheep=new HVolumeBrick16(scutmat, scvdpy,
					   "/opt/SCIRun/data/Geometry/volumes2/sheep-US.raw",
					   3, nworkers);
  InstanceWrapperObject *siw = new InstanceWrapperObject(sheep);

  Transform *strans = new Transform();
  strans->rotate(Vector(0,0,1), Vector(0,0,-1));
  //after rotation...
#define ORIGEX .256
#define ORIGEY .352
#define ORIGEZ .352
  //choose maximum from model, in this case y
#define SCALE EXTY/ORIGEY
  strans->pre_scale(Vector(SCALE,SCALE,SCALE));
  strans->pre_translate(Vector(CENX, CENY, CENZ));
#define ZMIN CENZ-(ORIGZ*SCALE)/2.0
#define LOWERZ 0.5+HFLT - ZMIN
  strans->pre_translate(Vector(0,0,LOWERZ)); //place at HFLT above surface
#undef ORIGEX
#undef ORIGEY
#undef ORIGEZ
#undef SCALE
#undef ZMIN
#undef LOWERZ
  /*
  strans->pre_scale(Vector(6.02,6.02,6.02)); //scale to fit max
  strans->pre_translate(Vector(-8, 8, 1.75));
  strans->pre_translate(Vector(0,0,0.30112)); //place 1cm above table
  */

  SpinningInstance *sinst = new SpinningInstance(siw, strans, Point(-8,8,1.56), Vector(0,0,1), 0.1);
  
  sinst->set_name("Spinning Sheep Heart");

  CutGroup *scut = new CutGroup(scpdpy, true);
  scut->add(sinst);
  scut->set_name("Sheep Heart Cutting Plane");
  sinst->addCPDpy(scpdpy);
#endif

#undef HFLT
#undef EXTZ
#undef EXTX
#undef EXTY
#undef CENX
#undef CENY
#undef CENZ

  //PUT THE VOLUMES INTO A SWITCHING GROUP  
  SelectableGroup *sg = new SelectableGroup(10);

#ifdef ADD_VIS_FEM
  BBox vinst_bbox;
  vinst->compute_bounds(vinst_bbox, 0);
  cerr << "Visible Female bbox="<<vinst_bbox.min()<<"-"<<vinst_bbox.max()<<"(diag="<<vinst_bbox.diagonal()<<"\n";
  sg->add(vcut);
#endif
#ifdef ADD_HEAD
  BBox hinst_bbox;
  hinst->compute_bounds(hinst_bbox, 0);
  cerr << "Head bbox="<<hinst_bbox.min()<<"-"<<hinst_bbox.max()<<"(diag="<<hinst_bbox.diagonal()<<"\n";
  sg->add(hcut);
#endif
#ifdef ADD_CSAFE_FIRE
  BBox fire_inst_bbox;
  fire_inst->compute_bounds(fire_inst_bbox, 0);
  cerr << "CSAFE Fire bbox="<<fire_inst_bbox.min()<<"-"<<fire_inst_bbox.max()<<"(diag="<<fire_inst_bbox.diagonal()<<"\n";
  sg->add(fire_inst);
#endif
#ifdef ADD_GEO_DATA
  BBox ginst_bbox;
  gcut->compute_bounds(ginst_bbox, 0);
  cerr << "Geoscience bbox="<<ginst_bbox.min()<<"-"<<ginst_bbox.max()<<"(diag="<<ginst_bbox.diagonal()<<"\n";
  sg->add(gcut);
#endif
#ifdef ADD_SHEEP
  BBox sinst_bbox;
  sinst->compute_bounds(sinst_bbox, 0);
  cerr << "Sheep Heart bbox="<<sinst_bbox.min()<<"-"<<sinst_bbox.max()<<"(diag="<<sinst_bbox.diagonal()<<"\n";
  sg->add(scut);
#endif
#ifdef ADD_DTIGLYPH
  BBox glyph_bbox;
  glyphs->compute_bounds(glyph_bbox, 0);
  cerr << "Glyph bbox="<<glyph_bbox.min()<<"-"<<glyph_bbox.max()<<"(diag="<<glyph_bbox.diagonal()<<"\n";
  sg->add(glyphs);
#endif

#ifdef ADD_BOX
  BBox boxg_bbox;
  boxg->compute_bounds(boxg_bbox, 0);
  cerr << "Box bbox="<<boxg_bbox.min()<<"-"<<boxg_bbox.max()<<"(diag="<<boxg_bbox.diagonal()<<"\n";
  sg->add(boxg);
#endif

#ifdef ADD_SPHERE
  BBox sphereg_bbox;
  sphereg->compute_bounds(sphereg_bbox, 0);
  cerr << "Sphere bbox="<<sphereg_bbox.min()<<"-"<<sphereg_bbox.max()<<"(diag="<<sphereg_bbox.diagonal()<<"\n";
  sg->add(sphereg);
//#else
//  g->add(sphereg);
#endif

  sg->set_name("VolVis Selection");
  BBox sg_bbox;
  sphereg->compute_bounds(sg_bbox, 0);
  cerr << "Selectable Group bbox="<<sg_bbox.min()<<"-"<<sg_bbox.max()<<"(diag="<<sg_bbox.diagonal()<<"\n";
  g->add(sg);

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);

  Scene *scene = new Scene(new Grid(g, 64),
			   cam, bgcolor, cdown, cup, groundplane, 0.3);

//  Scene *scene = new Scene(new Grid(g, 64),
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);
//  Scene *scene = new Scene(new BV1(g),
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);
//  Scene *scene = new Scene(new BV1(g),
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);

  scene->shadowobj = s;

//  Scene *scene = new Scene(new HierarchicalGrid(g, 8, 8, 8, 20, 20, 5),
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);
//  Scene *scene = new Scene(g,
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);

#ifdef ADD_BRICKBRACK
  scene->addObjectOfInterest( smw, true);  
#endif

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;
  scene->animate=true;

  scene->addObjectOfInterest( sg, true );

#ifdef ADD_VIS_FEM
  scene->addObjectOfInterest( vinst, false );
  scene->attach_auxiliary_display(vcvdpy);
  vcvdpy->setName("Visible Female Volume");
  scene->attach_display(vcvdpy);
  (new Thread(vcvdpy, "VFEM Volume Dpy"))->detach();

  scene->addObjectOfInterest( vcut, false );
  scene->attach_auxiliary_display(vcpdpy);
  vcpdpy->setName("Visible Female Cutting Plane");
  scene->attach_display(vcpdpy);
  (new Thread(vcpdpy, "VFEM CutPlane Dpy"))->detach();
#endif
#ifdef ADD_HEAD
  scene->addObjectOfInterest( hinst, false );
  scene->attach_auxiliary_display(hcvdpy);
  hcvdpy->setName("Brain Volume");
  scene->attach_display(hcvdpy);
  (new Thread(hcvdpy, "HEAD Volume Dpy"))->detach();

  scene->addObjectOfInterest( hcut, false );
  scene->attach_auxiliary_display(hcpdpy);
  hcpdpy->setName("Brain Cutting Plane");
  scene->attach_display(hcpdpy);
  (new Thread(hcpdpy, "VFEM CutPlane Dpy"))->detach();
#endif
#ifdef ADD_CSAFE_FIRE
  scene->addObjectOfInterest( fire_geom, false );
  scene->addObjectOfInterest( temp_field, true );
  scene->addObjectOfInterest( fire_inst, false );
  scene->addObjectOfInterest( temp_inst, true );
  //  scene->attach_auxiliary_display(fcdpy);
  //  fcdpy->setName("CSAFE Fire Temperature Transfer Function");
  //  scene->attach_display(fcdpy);
  //  (new Thread(fcdpy,"CSAFE Fire Temperature Transfer Function Dpy"))->detach();
  scene->attach_auxiliary_display(vel_dpy);
  vel_dpy->setName("CSAFE Fire Volume");
  scene->attach_display(vel_dpy);
  (new Thread(vel_dpy, "CSAFE Fire Volume Dpy"))->detach();
#endif
#ifdef ADD_GEO_DATA
  scene->addObjectOfInterest( ginst, false );
  scene->attach_auxiliary_display(gcvdpy);
  gcvdpy->setName("Geological Volume");
  scene->attach_display(gcvdpy);
  (new Thread(gcvdpy, "GEO Volume Dpy"))->detach();

  scene->addObjectOfInterest( gcut, false );
  scene->attach_auxiliary_display(gcpdpy);
  gcpdpy->setName("Geological Cutting Plane");
  scene->attach_display(gcpdpy);
  (new Thread(gcpdpy, "GEO CutPlane Dpy"))->detach();

#endif
#ifdef ADD_SHEEP
  scene->addObjectOfInterest( sinst, false );
  scene->attach_auxiliary_display(scvdpy);
  scvdpy->setName("Sheep Heart Volume");
  scene->attach_display(scvdpy);
  (new Thread(scvdpy, "SHEEP Heart Volume Dpy"))->detach();

  scene->addObjectOfInterest( scut, false );
  scene->attach_auxiliary_display(scpdpy);
  scpdpy->setName("Sheep Heart Cutting Plane");
  scene->attach_display(scpdpy);
  (new Thread(scpdpy, "SHEEP CutPlane Dpy"))->detach();
#endif

  for (l=0; l<holo_lights.size(); l++)
    scene->add_per_matl_light(holo_lights[l]);

  for (l=0; l<room_lights.size(); l++)
    scene->add_per_matl_light(room_lights[l]);
  
  return scene;
}

/*

./rtrt -np 10 -scene scenes/science-room.mo /opt/SCIRun/data/Geometry/volumes2/gk/gk2-rcc-b06-mask-s070.nhdr fa 0.45 0.55 -gridcellsize 100 -nl 2

*/
