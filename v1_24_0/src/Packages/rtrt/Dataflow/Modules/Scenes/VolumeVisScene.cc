/*
 *  VolumeVis.cc:  Scene for the Real Time Ray Tracer renderer
 *
 *  This module creates a scene for the real time ray tracer.
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

// SCIRun stuff
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Thread/Thread.h>
// rtrt Core stuff
#include <Packages/rtrt/Core/TimeObj.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/VolumeVis.h>
#include <Packages/rtrt/Core/VolumeVisDpy.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
// all the module stuff
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/rtrt/Dataflow/Ports/ScenePort.h>
// general libs
#include <nrrd.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <vector>
#include <float.h>
#include <time.h>
#include <stdlib.h>

namespace rtrt {

using namespace SCIRun;
using namespace std;

class VolumeVisScene : public Module {
public:
  VolumeVisScene(GuiContext *ctx);
  virtual ~VolumeVisScene();
  virtual void execute();
  void tcl_command(GuiArgs& args, void* userdata);

private:
  void create_dirs(Vector* objset);
  void create_objs(Group* group, const Point& center,
		   double radius, const Vector& dir, int depth,
		   Vector* objset, Material* matl);
  void make_box(Group* group, Material* matl, const Point& corner,
		const Vector& x, const Vector& y, const Vector& z);
  Object* make_obj(int size);
  Scene* make_scene();

  SceneOPort *scene_out_port;

  GuiInt scene_type_gui;
  GuiString data_file_gui;
  GuiInt do_phong_gui;
  GuiInt ncolors_gui;
  GuiDouble t_inc_gui;
  GuiDouble spec_coeff_gui;
  GuiDouble ambient_gui;
  GuiDouble diffuse_gui;
  GuiDouble specular_gui;
  GuiDouble val_gui;
  GuiInt override_data_min_gui;
  GuiInt override_data_max_gui;
  GuiDouble data_min_in_gui;
  GuiDouble data_max_in_gui;
  GuiDouble frame_rate_gui;

  // Pointers to data structures
  TimeObj *timeobj;
};

static string widget_name("VolumeVisScene Widget");

DECLARE_MAKER(VolumeVisScene)

VolumeVisScene::VolumeVisScene(GuiContext* ctx)
  : Module("VolumeVisScene", ctx, Filter, "Scenes", "rtrt"),
    scene_type_gui(ctx->subVar("scene_type_gui")),
    data_file_gui(ctx->subVar("data_file_gui")),
    do_phong_gui(ctx->subVar("do_phong_gui")),
    ncolors_gui(ctx->subVar("ncolors_gui")),
    t_inc_gui(ctx->subVar("t_inc_gui")),
    spec_coeff_gui(ctx->subVar("spec_coeff_gui")),
    ambient_gui(ctx->subVar("ambient_gui")),
    diffuse_gui(ctx->subVar("diffuse_gui")),
    specular_gui(ctx->subVar("specular_gui")),
    val_gui(ctx->subVar("val_gui")),
    override_data_min_gui(ctx->subVar("override_data_min_gui")),
    override_data_max_gui(ctx->subVar("override_data_max_gui")),
    data_min_in_gui(ctx->subVar("data_min_in_gui")),
    data_max_in_gui(ctx->subVar("data_max_in_gui")),
    frame_rate_gui(ctx->subVar("frame_rate_gui")),
    timeobj(0)
{
  //  inColorMap = scinew ColorMapIPort( this, "ColorMap",
  //				     ColorMapIPort::Atomic);
  //  add_iport( inColorMap);
}

VolumeVisScene::~VolumeVisScene()
{
}

void VolumeVisScene::execute()
{
  reset_vars();
  // Create the output port
  scene_out_port = (SceneOPort *) get_oport("Scene");
  Scene * scene = make_scene();
  SceneContainer *container = scinew SceneContainer();
  container->put_scene(scene);
  scene_out_port->send( container );
}

// This is called when the tcl code explicity calls a function besides
// needexecute.
void VolumeVisScene::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("VolumeVisScene needs a minor command");
    return;
  } else if (args[1] == "rate_chage") {
    if (timeobj != 0) {
      timeobj->change_rate(frame_rate_gui.get());
    }
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

/////////////////////////////////////////////////////////
// stuff to make the scene

static void get_material(Array1<Color> &matls, Array1<AlphaPos> &alphas) {

  matls.add(Color(0,0,1));
  matls.add(Color(0,0.4,1));
  matls.add(Color(0,0.8,1));
  matls.add(Color(0,1,0.8));
  matls.add(Color(0,1,0.4));
  matls.add(Color(0,1,0));
  matls.add(Color(0.4,1,0));
  matls.add(Color(0.8,1,0));
  matls.add(Color(1,0.9176,0));
  matls.add(Color(1,0.8,0));
  matls.add(Color(1,0.4,0));
  matls.add(Color(1,0,0));

  alphas.add(AlphaPos(0       , 0));  // pos 0
  alphas.add(AlphaPos(28.0/255, 0));  // pos 28
  alphas.add(AlphaPos(64.0/255, 0.1));// pos 64
  alphas.add(AlphaPos(100.0/255,0));  // pos 100
  alphas.add(AlphaPos(156.0/255,0));  // pos 156
  alphas.add(AlphaPos(192.0/255,1));  // pos 192
  alphas.add(AlphaPos(228.0/255,0));  // pos 192 
  alphas.add(AlphaPos(1,        0));  // pos 192
}

static void get_material2(Array1<Color> &matls, Array1<AlphaPos> &alphas) {

  float div = 1.0/255;
  matls.add(Color(255, 255, 255) * div);
  matls.add(Color(255, 255, 180) * div);
  matls.add(Color(255, 247, 120) * div);   
  matls.add(Color(255, 228, 80) * div);
  matls.add(Color(255, 204, 55) * div);   
  matls.add(Color(255, 163, 20) * div);
  matls.add(Color(255, 120, 0) * div);   
  matls.add(Color(230, 71, 0) * div);
  matls.add(Color(200, 41, 0) * div);   
  matls.add(Color(153, 18, 0) * div);
  matls.add(Color(102, 2, 0) * div);   
  matls.add(Color(52, 0, 0) * div);
  matls.add(Color(0, 0, 0) * div);


  alphas.add(AlphaPos(0       , 0));  // pos 0
  alphas.add(AlphaPos(28.0/255, 0));  // pos 28
  alphas.add(AlphaPos(64.0/255, 0.1));// pos 64
  alphas.add(AlphaPos(100.0/255,0));  // pos 100
  alphas.add(AlphaPos(156.0/255,0));  // pos 156
  alphas.add(AlphaPos(192.0/255,1));  // pos 192
  alphas.add(AlphaPos(228.0/255,0));  // pos 192 
  alphas.add(AlphaPos(1,        0));  // pos 192
}

static VolumeVis *create_volume_from_nrrd(char *filename,
				   bool override_data_min, double data_min_in,
				   bool override_data_max, double data_max_in,
				   double spec_coeff, double ambient,
				   double diffuse, double specular,
				   VolumeVisDpy *dpy)
{
  BrickArray3<float> data;
  float data_min = FLT_MAX;
  float data_max = -FLT_MAX;
  Point minP, maxP;
  // Do the nrrd stuff
  Nrrd *n = nrrdNew();
  // load the nrrd in
  cout << "VolumeVisScene::create_volume_from_nrrd::Loading "<<filename<<endl;
  if (nrrdLoad(n,filename)) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd "<< filename <<": "<<err<<"\n";
    free(err);
    biffDone(NRRD);
    return 0;
  }
  cout << "VolumeVisScene::create_volume_from_nrrd::"<<filename<<" loaded\n";
  // check to make sure the dimensions are good
  if (n->dim != 3) {
    cerr << "VolumeVisMod error: nrrd->dim="<<n->dim<<"\n";
    cerr << "  Can only deal with 3-dimensional scalar fields... sorry.\n";
    return 0;
  }
  // convert the type to floats if you need to
  nrrdBigInt num_elements = nrrdElementNumber(n);
  cerr << "Number of data members = " << num_elements << endl;
  if (n->type != nrrdTypeFloat) {
    cerr << "Converting type from ";
    switch(n->type) {
    case nrrdTypeUnknown: cerr << "nrrdTypeUnknown"; break;
    case nrrdTypeChar: cerr << "nrrdTypeChar"; break;
    case nrrdTypeUChar: cerr << "nrrdTypeUChar"; break;
    case nrrdTypeShort: cerr << "nrrdTypeShort"; break;
    case nrrdTypeUShort: cerr << "nrrdTypeUShort"; break;
    case nrrdTypeInt: cerr << "nrrdTypeInt"; break;
    case nrrdTypeUInt: cerr << "nrrdTypeUInt"; break;
    case nrrdTypeLLong: cerr << "nrrdTypeLLong"; break;
    case nrrdTypeULLong: cerr << "nrrdTypeULLong"; break;
    case nrrdTypeDouble: cerr << "nrrdTypeDouble"; break;
    default: cerr << "Unknown!!";
    }
    cerr << " to nrrdTypeFloat\n";
    Nrrd *new_n = nrrdNew();
    nrrdConvert(new_n, n, nrrdTypeFloat);
    // since the data was copied blow away the memory for the old nrrd
    nrrdNuke(n);
    n = new_n;
    cerr << "Number of data members = " << num_elements << endl;
  }
  // get the dimensions
  int nx, ny, nz;
  nx = n->axis[0].size;
  ny = n->axis[1].size;
  nz = n->axis[2].size;
  cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
  cout << "total = " << nz * ny * nz << endl;
  cout << "spacing = " << n->axis[0].spacing << " x "<<n->axis[1].spacing<< " x "<<n->axis[2].spacing<< endl;
  data.resize(nx,ny,nz); // resize the bricked data
  // get the physical bounds
  minP = Point(0,0,0);
  maxP = Point((nx - 1) * n->axis[0].spacing,
	       (ny - 1) * n->axis[1].spacing,
	       (nz - 1) * n->axis[2].spacing);
  // lets normalize the dimensions to 1
  Vector size = maxP - minP;
  // find the biggest dimension
  double max_dim = Max(Max(size.x(),size.y()),size.z());
  maxP = ((maxP-minP)/max_dim).asPoint();
  minP = Point(0,0,0);
  // copy the data into the brickArray
  cerr << "Number of data members = " << num_elements << endl;
  float *p = (float*)n->data; // get the pointer to the raw data
  for (int z = 0; z < nz; z++)
    for (int y = 0; y < ny; y++)
      for (int x = 0; x < nx; x++) {
	float val = *p++;
	data(x,y,z) = val;
	// also find the min and max
	if (val < data_min)
	  data_min = val;
	else if (val > data_max)
	  data_max = val;
      }
#if 0
  // compute the min and max of the data
  double dmin,dmax;
  nrrdMinMaxFind(&dmin,&dmax,n);
  data_min = (float)dmin;
  data_max = (float)dmax;
#endif
  // delete the memory that is no longer in use
  nrrdNuke(n);


  // override the min and max if it was passed in
  if (override_data_min)
    data_min = data_min_in;
  if (override_data_max)
    data_max = data_max_in;

  cout << "minP = "<<minP<<", maxP = "<<maxP<<endl;

  return new VolumeVis(data, data_min, data_max,
		       nx, ny, nz,
		       minP, maxP,
		       spec_coeff, ambient, diffuse,
		       specular, dpy);
}  

static VolumeVis *create_volume_default(int scene_type, double val,
				 int nx, int ny, int nz,
				 bool override_data_min, double data_min_in,
				 bool override_data_max, double data_max_in,
				 double spec_coeff, double ambient,
				 double diffuse, double specular,
				 VolumeVisDpy *dpy)
{
  BrickArray3<float> data;

  // make sure dimensions are good
  if (!nx) nx = 1;
  if (!ny) ny = 1;
  if (!nz) nz = 1;
  // resize the bricked data
  data.resize(nx,ny,nz);
  // fill the data with values
  for (int x = 0; x < nx; x++) {
    for (int y = 0; y < ny; y++) {
      for (int z = 0; z < nz; z++) {
	switch (scene_type) {
	case 0:
	  data(x,y,z) = z;
	  break;
	case 1:
	  data(x,y,z) = y*z;
	  break;
	case 2:
	  data(x,y,z) = x*y*z;
	  break;
	case 3:
	  data(x,y,z) = y+z;
	  break;
	case 4:
	  data(x,y,z) = x+y+z;
	  break;
	case 5:
	  data(x,y,z) = val;
	  break;
	}
      }
    }
  }
  // compute the min and max of the data
  float data_min = 0;
  float data_max;
  switch (scene_type) {
  case 0:
    data_max = (nz-1);
    break;
  case 1:
    data_max = (ny-1) * (nz-1);
    break;
  case 2:
    data_max = (nx-1) * (ny-1) * (nz-1);
    break;
  case 3:
    data_max = (ny-1) + (nz-1);
    break;
  case 4:
    data_max = (nx-1) + (ny-1) + (nz-1);
    break;
  case 5:
    data_min = val;
    data_max = val;
    break;
  }
  // set the physical dimensions of the data
  Point minP(0,0,0), maxP(1,1,1);
  cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
  cout << "total = " << nz * ny * nz << endl;

  cout << "minP = "<<minP<<", maxP = "<<maxP<<endl;

  // override the min and max if it was passed in
  if (override_data_min)
    data_min = data_min_in;
  if (override_data_max)
    data_max = data_max_in;

  return new VolumeVis(data, data_min, data_max,
		       nx, ny, nz,
		       minP, maxP,
		       spec_coeff, ambient, diffuse,
		       specular, dpy);
}

Scene* VolumeVisScene::make_scene()
{
  int nx = 20;
  int ny = 30;
  int nz = 40;
  int scene_type = scene_type_gui.get();
  vector<string> data_files;
  string data_file(data_file_gui.get());
  bool cut=false;
  bool do_phong = do_phong_gui.get() != 0;
  int ncolors=256;
  float t_inc = t_inc_gui.get(); // 0.01;
  double spec_coeff = spec_coeff_gui.get(); //64;
  double ambient = ambient_gui.get(); // 0.5;
  double diffuse = diffuse_gui.get(); // 1.0;
  double specular = specular_gui.get(); // 1.0;
  float val = val_gui.get(); // 1;
  bool override_data_min = override_data_min_gui.get() != 0;
  bool override_data_max = override_data_max_gui.get() != 0;
  float data_min_in = data_min_in_gui.get();
  float data_max_in = data_max_in_gui.get();;
  float frame_rate = frame_rate_gui.get();
  
  switch (scene_type) {
  case 6:
    data_files.push_back(data_file);
    break;
  case 7:
    scene_type = 6;
    // open up the file and then suck in all the files
    cout << "Reading nrrd file list from " << data_file << endl;
    ifstream in(data_file.c_str());
    while (in) {
      string file;
      in >> file;
      data_files.push_back(file);
      cout << "Nrrd file: "<<file<<"\n";
    }
    cout << "Read "<<data_files.size()<<" nrrd file names.\n";
    break;
  }
  
  // check the parameters
  if (scene_type == 6)
    if (data_files.size() == 0)
      // the file was not set
      scene_type = 0;
  
  Camera cam(Point(0.5,0.5,3), Point(0.5,0.5,0.5),
	     Vector(0,1,0), 40.0);
  
  double ambient_scale=1.0;
  
  //  double bgscale=0.5;
  //  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  Color bgcolor(1.,1.,1.);
  
  Array1<Color> matls;
  Array1<AlphaPos> alphas;
  get_material2(matls,alphas);
  VolumeVisDpy *dpy = new VolumeVisDpy(matls, alphas, ncolors, t_inc);

  // Generate the data
  Object *obj;

  if (scene_type == 6) {
    if (data_files.size() > 1) {
      // create a timeobj
      timeobj = new TimeObj(frame_rate);
      for(unsigned int i = 0; i < data_files.size(); i++) {
	char *myfile = strdup(data_files[i].c_str());
	Object *volume = (Object*)create_volume_from_nrrd
	  (myfile, override_data_min, data_min_in,
	   override_data_max, data_max_in, spec_coeff,
	   ambient, diffuse, specular, dpy);
	// add the volume to the timeobj
	if (myfile)
	  free(myfile);
	if (volume == 0)
	  // there was a problem
	  continue;
	timeobj->add(volume);
      }
      obj = (Object*)timeobj;
    } else {
      char *myfile = strdup(data_files[0].c_str());
      obj = (Object*)create_volume_from_nrrd
	(myfile, override_data_min, data_min_in,
	 override_data_max, data_max_in, spec_coeff,
	 ambient, diffuse, specular, dpy);
      if (myfile)
	free(myfile);
    }
  } else {
    obj = (Object*) create_volume_default
	(scene_type, val, nx, ny, nz, override_data_min, data_min_in,
	 override_data_max, data_max_in, spec_coeff, ambient,
	 diffuse, specular, dpy);
  }

  new Thread(dpy, "VolumeVis display thread");
  
  if(cut){
    PlaneDpy* pd=new PlaneDpy(Vector(0,0,1), Point(0,0,0));
    obj=(Object*)new CutPlane(obj, pd);
    new Thread(pd, "Cutting plane display thread");
  }
  Group* all = new Group();
  all->add(obj);

  Plane groundplane ( Point(-500, 300, 0), Vector(7, -3, 2) );
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);

  Scene* scene=new Scene(all, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);
  
  scene->add_light(new Light(Point(500,-300,300), Color(.8,.8,.8), 0));
  scene->shadow_mode=1;
  return scene;
}

} // End namespace rtrt

