#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/VolumeVisRGBA.h>
#include <Packages/rtrt/Core/VolumeVisDpy.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Core/Thread/Thread.h>
#include <teem/nrrd.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <vector>
#include <string>

using namespace std;
using namespace rtrt;
using SCIRun::Thread;

// Whether or not to use the HVolumeVis code
//static bool use_hvolume = false;
//static int np = 1;
//static int depth = 3;

VolumeVisBase *create_volume_from_nrrd(char *filename,
				   bool override_data_min, double data_min_in,
				   bool override_data_max, double data_max_in,
				   double spec_coeff, double ambient,
				   double diffuse, double specular,
				   VolumeVisDpy *dpy)
{
  BrickArray3<Color_floatA> data;
  float data_min = FLT_MAX;
  float data_max = -FLT_MAX;
  Point minP, maxP;
  // Do the nrrd stuff
  Nrrd *n = nrrdNew();
  // load the nrrd in
  if (nrrdLoad(n,filename,NULL)) {
    char *err = biffGet(NRRD);
    cerr << "Error reading nrrd "<< filename <<": "<<err<<"\n";
    free(err);
    biffDone(NRRD);
    return 0;
  }
  // check to make sure the dimensions are good
  if (n->dim != 4) {
    cerr << "VolumeVisMod error: nrrd->dim="<<n->dim<<"\n";
    cerr << "  Can only deal with 3-dimensional RGBA fields... sorry.\n";
    return 0;
  }
  if ( n->axis[0].size != 4 ) {
    cerr << "Can only handle RGBA data.\n";
    return 0;
  }
  size_t num_elements = nrrdElementNumber(n);
  cerr << "Number of data members = " << num_elements << endl;
  // convert the type to floats if you need to
  if (n->type != nrrdTypeFloat) {
    cerr << "I only know how to deal with type float!!\n";
    cerr << "Type is :";
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
    return 0;
  }
  cerr << "Number of data members = " << num_elements << endl;
  // get the dimensions
  int nx, ny, nz;
  nx = n->axis[1].size;
  ny = n->axis[2].size;
  nz = n->axis[3].size;
  cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
  cout << "total = " << nz * ny * nz << endl;
  cout << "spacing = " << n->axis[0].spacing << " x "<<n->axis[1].spacing<< " x "<<n->axis[2].spacing<< endl;
  for (int i = 0; i<n->dim; i++)
    if (!(AIR_EXISTS(n->axis[i].spacing))) {
      cout <<"spacing for axis "<<i<<" does not exist.  Setting to 1.\n";
      n->axis[i].spacing = 1;
    }
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
	float r = *p++;
	float g = *p++;
	float b = *p++;
	float a = *p++;
	data(x,y,z) = Color_floatA(Color(r,g,b), a);
	// also find the min and max
	if (a < data_min)
	  data_min = a;
	else if (a > data_max)
	  data_max = a;
      }
  // delete the memory that is no longer in use
  nrrdNuke(n);


  // override the min and max if it was passed in
  if (override_data_min)
    data_min = data_min_in;
  if (override_data_max)
    data_max = data_max_in;

  cout << "minP = "<<minP<<", maxP = "<<maxP<<endl;

  return new VolumeVisRGBA<Color_floatA>(data, data_min, data_max,
					 nx, ny, nz,
					 minP, maxP,
					 spec_coeff, ambient, diffuse,
					 specular, dpy);
}  

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  int scene_type = 0;
  vector<string> data_files;
  bool cut=false;
  float t_inc = 0.01;
  double spec_coeff = 64;
  double ambient = 0.5;
  double diffuse = 1.0;
  double specular = 1.0;
  bool override_data_min = false;
  bool override_data_max = false;
  float data_min_in = -1;
  float data_max_in = -1;
  float frame_rate = 3;
  Color bgcolor(0,0,0);
  
  for(int i=1;i<argc;i++){
    if (strcmp(argv[i], "-type")==0) {
      i++;
      if (strcmp(argv[i], "nrrd") == 0) {
	scene_type = 6;
	i++;
	data_files.push_back(argv[i]);
      } else if (strcmp(argv[i], "nrrdlist") == 0) {
	scene_type = 6;
	// open up the file and then suck in all the files
	i++;
	cout << "Reading nrrd file list from " << argv[i] << endl;
	ifstream in(argv[i]);
	while (in) {
	  string file;
	  in >> file;
	  data_files.push_back(file);
	  cout << "Nrrd file: "<<file<<"\n";
	}
	cout << "Read "<<data_files.size()<<" nrrd files.\n";
      }
    } else if(strcmp(argv[i], "-cut")==0){
      cut=true;
    } else if(strcmp(argv[i], "-tinc")==0){
      i++;
      t_inc = atof(argv[i]);
    } else if(strcmp(argv[i], "-min")==0){
      i++;
      data_min_in = atof(argv[i]);
      override_data_min = true;
    } else if(strcmp(argv[i], "-max")==0){
      i++;
      data_max_in = atof(argv[i]);
      override_data_max = true;
    } else if(strcmp(argv[i], "-spow")==0){
      i++;
      spec_coeff = atof(argv[i]);
    } else if(strcmp(argv[i], "-ambient")==0){
      i++;
      ambient = atof(argv[i]);
    } else if(strcmp(argv[i], "-diffuse")==0){
      i++;
      diffuse = atof(argv[i]);
    } else if(strcmp(argv[i], "-specular")==0){
      i++;
      specular = atof(argv[i]);
    } else if(strcmp(argv[i], "-rate")==0){
      i++;
      frame_rate = atof(argv[i]);
    } else if(strcmp(argv[i], "-bgcolor")==0){
      float r,g,b;
      r = atof(argv[++i]);
      g = atof(argv[++i]);
      b = atof(argv[++i]);
      bgcolor = Color(r,g,b);
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      cerr << "Valid options for scene: " << argv[0] << '\n';
      cerr << " -type [int or \"nrrd\"]\n";
      cerr << "\t\tnrrd [path to nrrd file]\n";
      cerr << "\t\tnrrdlist [path to file with name of nrrds]\n";
      cerr << " -cut - turn on the cutting plane\n";
      cerr << " -tinc [float] - the number of samples per unit\n";
      cerr << " -spow [float] - the spectral exponent\n";
      cerr << " -ambient [float] - the ambient factor\n";
      cerr << " -diffuse [float] - the diffuse factor\n";
      cerr << " -specular [float] - the specular factor\n";
      cerr << " -rate [float] - frame rate of the time steps\n";
      cerr << " -bgcolor [float] [float] [float] - the three floats are r, g, b\n";
      return 0;
    }
  }

  // check the parameters
  if (scene_type == 6)
    if (data_files.size() == 0)
      // the file was not set
      scene_type = 0;
  
  Camera cam(Point(0.5,0.5,3), Point(0.5,0.5,0.5),
	     Vector(0,1,0), 40.0);

  Array1<Color> matls;
  matls.add(Color(1,0,1));
  matls.add(Color(0,1,1));
  Array1<AlphaPos> alphas;
  alphas.add(AlphaPos(0,1));
  alphas.add(AlphaPos(1,1));
  
  VolumeVisDpy *dpy = new VolumeVisDpy(matls, alphas, 10, t_inc);

  // Generate the data
  Object *obj;

  if (scene_type == 6) {
    if (data_files.size() > 1) {
      // create a timeobj
      //      TimeObj *timeobj = new TimeObj(frame_rate);
      SelectableGroup *timeobj = new SelectableGroup(1.0/frame_rate);
      timeobj->set_name("Volume Data");
      for(unsigned int i = 0; i < data_files.size(); i++) {
	char *myfile = strdup(data_files[i].c_str());
	Object *volume = (Object*)create_volume_from_nrrd
	  (myfile, override_data_min, data_min_in,
	   override_data_max, data_max_in, spec_coeff,
	   ambient, diffuse, specular, dpy);
	if (volume == 0) {
	  return 0;
	}
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
      if (obj == 0)
	return 0;
      if (myfile)
	free(myfile);
    }
  } else {
    cerr << "Unknown scene type, try nrrd or nrrdlist\n";
    return 0;
  }

  (new Thread(dpy, "VolumeVis display thread"))->detach();

  if(cut){
    PlaneDpy* pd=new PlaneDpy(Vector(0,0,1), Point(0,0,0));
    obj=(Object*)new CutPlane(obj, pd);
    (new Thread(pd, "Cutting plane display thread"))->detach();
  }
  Group* all = new Group();
  all->add(obj);

  Plane groundplane ( Point(-500, 300, 0), Vector(7, -3, 2) );
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);

  double ambient_scale=1.0;
  
  Scene* scene=new Scene(all, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);

  Light *light0 = new Light(Point(500,-300,300), Color(.8,.8,.8), 0);
  light0->name_ = "light 0";
  scene->add_light(light0);
  //scene->shadow_mode=1;
  scene->attach_display(dpy);
  scene->addObjectOfInterest(obj,true);
  
  return scene;
}



