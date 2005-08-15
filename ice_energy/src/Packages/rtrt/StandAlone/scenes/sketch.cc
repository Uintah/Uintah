#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/HVolumeBrickColor.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/SketchMaterial.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>

using namespace rtrt;
using SCIRun::Thread;

bool group_in_timeobj = true;

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  char *me = argv[0];
  char *errS;
  int depth=3;
  Array1<char*> files;
  bool cut=false;
  double rate=3;
  Array1<char*> sil_filenames;
  Array1<Material*> gui_materials;
  char *colormap_file = 0;
  
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-depth")==0){
      i++;
      depth=atoi(argv[i]);
    } else if(strcmp(argv[i], "-cut")==0){
      cut=true;
    } else if(strcmp(argv[i], "-rate")==0){
      rate = atof(argv[++i]);
    } else if(strcmp(argv[i], "-sil")==0){
      sil_filenames.add(argv[++i]);
    } else if(strcmp(argv[i], "-cmap")==0){
      colormap_file = argv[++i];
    } else if(argv[i][0] != '-'){
      files.add(argv[i]);
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      cerr << "Valid options for scene: " << argv[0] << '\n';
      cerr << " -depth n   - set depth of hierarchy\n";
      cerr << " file       - raw file name\n";
      cerr << " -cmap [file.cmp] - file to use for a colormap\n";
      return 0;
    }
  }
  
  if(files.size()==0){
    cerr << "Must specify at least one file\n";
    return 0;
  }
  if(files.size() != sil_filenames.size()) {
    cerr << "Must have the same number of files as silhouette files\n";
    return 0;
  }
  
  Camera cam(Point(1,3,1), Point(0.5,0.5,0.5), Vector(0,1,0), 40);
  
  Material* matl0;
  matl0=new Phong(Color(.6,1,.4), Color(0,0,0), 100, 0);
  
  VolumeDpy* dpy=new VolumeDpy(40);
  Object* obj;
  
  if(files.size()==1){
    HVolume<unsigned short, BrickArray3<unsigned short>, BrickArray3<VMCell<unsigned short> > > *hvol = new HVolume<unsigned short, BrickArray3<unsigned short>, BrickArray3<VMCell<unsigned short> > > (matl0, dpy, files[0], depth, nworkers);
#if 1
    // Now to add a new material
    BBox bbox;
    hvol->compute_bounds(bbox, 0);
    // Load in the silhouette transfer function for use in the texture.
    Nrrd *nrrdsil = nrrdNew();
    if (nrrdLoad(nrrdsil, sil_filenames[0],0)) {
      fprintf(stderr, "%s: problem with loading silhouette transfer function:\n%s\n",
	      me, errS = biffGetDone(NRRD));
      free(errS);
      return 0;
    }
    // Now copy the data over to an Array2
    if (nrrdsil->dim == 3) {
      // Need to project the data to only two dimensions.  The
      // first one is color, so we can just take the average of
      // it.  We also need to make sure they are floats.
      Nrrd *project = nrrdNew();
      nrrdProject(project, nrrdsil, 0, nrrdMeasureMean, nrrdTypeFloat);
      nrrdNuke(nrrdsil);
      nrrdsil = project;
    } else if (nrrdsil->dim != 2) {
      cerr << me << ":Don't know how to deal with a nrrd of dim "<<nrrdsil->dim << "\n";
      return 0;
    }
    if (nrrdsil->type != nrrdTypeFloat) {
      Nrrd *new_nrrd = nrrdNew();
      nrrdConvert(new_nrrd, nrrdsil, nrrdTypeFloat);
      // since the data was copied blow away the memory for the old nrrd
      nrrdNuke(nrrdsil);
      nrrdsil = new_nrrd;
    }
    int silx = nrrdsil->axis[0].size;
    int sily = nrrdsil->axis[1].size;
    Array2<float> sil_trans(silx,sily);
    float *data = (float*)(nrrdsil->data);
    for(int y = 0; y < sily; y++)
      for(int x = 0; x < silx; x++)
	{
	  sil_trans(x,y) = *data;
	  data++;
	}

    // Load the colormap
    Nrrd *nrrdcmap = nrrdNew();
    ScalarTransform1D<float, Color> *cmap;
    if (colormap_file) {
      if (nrrdLoad(nrrdcmap, colormap_file,0)) {
	fprintf(stderr, "%s: problem with loading cool2warm transfer function (%s):\n%s\n",
		me, colormap_file, errS = biffGetDone(NRRD));
	free(errS);
	return 0;
      }
      // Do some double checking
      if (nrrdcmap->dim != 2) {
	fprintf(stderr, "%s: colormap is not of dim 2 (actual = %d)\n", me,
		nrrdcmap->dim);
	return 0;
      }
      if (nrrdcmap->axis[0].size != 3) {
	fprintf(stderr, "%s: Only know how to deal with 3 component colors (actual = %d)\n", me,
		nrrdcmap->axis[0].size);
	return 0;
      }
      if (nrrdcmap->type != nrrdTypeFloat) {
	Nrrd *new_nrrd = nrrdNew();
	nrrdConvert(new_nrrd, nrrdcmap, nrrdTypeFloat);
	// since the data was copied blow away the memory for the old nrrd
	nrrdNuke(nrrdcmap);
	nrrdcmap = new_nrrd;
      }
      // Now that we have the colors load them into the colormap
      Array1<Color> colors(nrrdcmap->axis[1].size);
      float *data = (float*)(nrrdcmap->data);
      for(int i = 0; i < nrrdcmap->axis[1].size; i++)
	colors[i] = Color(*data++, *data++, *data++);
      cmap = new ScalarTransform1D<float, Color>(colors);
      if (AIR_EXISTS_D(nrrdcmap->axis[1].min) &&
	  AIR_EXISTS_D(nrrdcmap->axis[1].max))
	cmap->scale(nrrdcmap->axis[1].min, nrrdcmap->axis[1].max);
      else
	cmap->scale(-1,1);
    } else {
      Array1<ColorCell> ccells;
      ccells.add(ColorCell(Color(1.0980393, 1.0392157, 0.86274511), 0));
      ccells.add(ColorCell(Color(0.98039216, 0.74509805, 0.47058824), 0.6));
      ccells.add(ColorCell(Color(0.84313726, 0.54901963, 0.3137255), 1.5));
      ccells.add(ColorCell(Color(0.66666669, 0.49019608, 0.54901963), 2.5));
      ccells.add(ColorCell(Color(0.43137255, 0.3137255, 0.47058824), 3.7));
      ccells.add(ColorCell(Color(0.3137255, 0.27450982, 0.35294119), 5));
      ColorMap cm(ccells);
      cmap = new ScalarTransform1D<float, Color>(cm.slices.get_results_ref());
      cmap->scale(-1,1);
    }
    

    Material *sm = new SketchMaterial<BrickArray3<unsigned short>, unsigned short>(hvol->blockdata, bbox, sil_trans, nrrdsil->axis[0].max, cmap);
    hvol->set_matl(sm);
    gui_materials.add(sm);
#endif
    obj = hvol;
  } else {
    Group *group = 0;
    if (group_in_timeobj) {
      group = new SelectableGroup(1/rate);
      cout << "using time changing objects\n";
    } else {
      group=new Group();
      cout << "Grouping all objects together\n";
    }
    obj=group;
    for(int i=0;i<files.size();i++){
      Object* hvol = new HVolume<unsigned short, BrickArray3<unsigned short>, BrickArray3<VMCell<unsigned short> > > (matl0, dpy, files[i], depth, nworkers);
      group->add(hvol);
    }
  }

  if(cut){
    PlaneDpy* pd=new PlaneDpy(Vector(0,0,1), Point(0,0,100));
    obj=new CutPlane(obj, pd);
    obj->set_matl(matl0);
    (new Thread(pd, "Cutting plane display thread"))->detach();
  }

  // Start up the thread to handle the slider
  (new Thread(dpy, "Volume GUI thread"))->detach();
	
  //double bgscale=0.5;
  double ambient_scale=.5;

  Color bgcolor(0.01, 0.05, 0.3);
  Color cup(1, 0, 0);
  Color cdown(0, 0, 0.2);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, -1, 0) );
  Scene* scene=new Scene(obj, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale);
  scene->addAnimateObject(obj);
  for(int i = 0; i < gui_materials.size(); i++)
    scene->addGuiMaterial(gui_materials[i]);

  // Compute the location of the light
  BBox bounds;
  obj->compute_bounds(bounds, 0);
  // Take the corners and then extend one of them
  Point light_loc = bounds.max() + (bounds.max()-bounds.min())*0.2;
  
  Light *light = new Light(light_loc, Color(.8, .8, .8),
			   (bounds.max()-bounds.min()).length()*0.01);
  light->name_ = "Main Light";
  scene->add_light(light);

  // Change the background to LinearBackground
  scene->set_background_ptr( new LinearBackground(
						  Color(0.2, 0.4, 0.9),
						  Color(0.0,0.0,0.0),
						  Vector(1, 0, 0)) );

  scene->select_shadow_mode( No_Shadows );
  scene->attach_display(dpy);
  return scene;
}

