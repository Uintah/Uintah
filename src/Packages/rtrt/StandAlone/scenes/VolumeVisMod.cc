#include <Packages/rtrt/Core/Array1.cc>
#include <Packages/rtrt/Core/BrickArray3.cc>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/VolumeVis.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Core/Thread/Thread.h>
#include <nrrd.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace rtrt;
using SCIRun::Thread;

void get_material(Array1<Material*> &matls, Array1<float> &alphas,
		  bool do_phong) {
  CatmullRomSpline<Color> spline(0);
#if 1
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0,1));
  spline.add(Color(0,0.4,1));
  spline.add(Color(0,0.8,1));
  spline.add(Color(0,1,0.8));
  spline.add(Color(0,1,0.4));
  spline.add(Color(0,1,0));
  spline.add(Color(0.4,1,0));
  spline.add(Color(0.8,1,0));
  spline.add(Color(1,0.9176,0));
  spline.add(Color(1,0.8,0));
  spline.add(Color(1,0.4,0));
  spline.add(Color(1,0,0));
#else
  spline.add(Color(.4,.4,.4));
  spline.add(Color(.4,.4,1));
  //    for(int i=0;i<2;i++)
  spline.add(Color(.4,1,1));
  spline.add(Color(.4,1,.4));
  //    for(int i=0;i<3;i++)
  spline.add(Color(1,1,.4));
  //    for(int i=0;i<300;i++)
  spline.add(Color(1,.4,.4));
  spline.add(Color(1,.4,.4));
#endif
  CatmullRomSpline<float> alpha_spline(0);
#if 0
  alpha_spline.add(0.9);
  alpha_spline.add(0.9);
  alpha_spline.add(0.3);
  alpha_spline.add(0.1);
  alpha_spline.add(0.01);
  alpha_spline.add(0.001);
  alpha_spline.add(0.003);
  alpha_spline.add(0.005);
  alpha_spline.add(0.001);
  alpha_spline.add(0.01);
  alpha_spline.add(0.1);
#else
#if 1
  alpha_spline.add(0);
  alpha_spline.add(0);
  alpha_spline.add(0);
  //  alpha_spline.add(0.001);
  alpha_spline.add(1);
  alpha_spline.add(1);
#else
  alpha_spline.add(0);
  alpha_spline.add(0);
  alpha_spline.add(0);
  alpha_spline.add(0.03);
  alpha_spline.add(0.04);
  alpha_spline.add(0.05);
#endif
  //  alpha_spline.add(0.1);
#endif
  int ncolors=5000;
  matls.resize(ncolors);
  alphas.resize(ncolors);
  if (do_phong) {
    float Ka=1;
    float Kd=1;
    float Ks=1;
    float refl=0;
    float specpow=200;
    for(int i=0;i<ncolors;i++){
      float frac=float(i)/(ncolors-1);
      Color c(spline(frac));
      matls[i]=new Phong(c*Ka, c*Kd, c*Ks, specpow, refl);
      alphas[i] = alpha_spline(frac);
    }
  } else {
    for(int i=0;i<ncolors;i++){
      float frac=float(i)/(ncolors-1);
      Color c(spline(frac));
      matls[i]=new LambertianMaterial(c);
      alphas[i] = alpha_spline(frac);
      //cerr << ", alpha = " << alphas[i];
      //matls[i]=new LambertianMaterial(c*Kd);
    }
  }
}

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  int nx = 20;
  int ny = 30;
  int nz = 40;
  int scene_type = 0;
  char *nrrd_file = 0;
  bool cut=false;
  bool do_phong = true;
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-dim")==0){
      i++;
      if(sscanf(argv[i], "%dx%dx%d", &nx, &ny, &nz) != 3){
	cerr << "Error parsing dimensions: " << argv[i] << '\n';
	cerr << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
	exit(1);
      } 
    }
    else if (strcmp(argv[i], "-type")==0) {
      i++;
      if (strcmp(argv[i], "nrrd") == 0) {
	scene_type = 5;
	i++;
	nrrd_file = argv[i];
      } else {
	scene_type = atoi(argv[i]);
      }
    } else if(strcmp(argv[i], "-cut")==0){
      cut=true;
    } else if(strcmp(argv[i], "-lam")==0){
      do_phong = false;
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      cerr << "Valid options for scene: " << argv[0] << '\n';
      cerr << " -dim [int]x[int]x[int]";
      return 0;
    }
  }

  // check the parameters
  if (scene_type == 5)
    if (nrrd_file == 0)
      // the file was not set
      scene_type = 0;
  
  Camera cam(Point(0,400,0), Point(0,0,0),
	     Vector(0,1,0), 60.0);
  
  double bgscale=0.5;
  double ambient_scale=1.0;
  
  //  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  Color bgcolor(0.,0.,0.);
  
  Group* all = new Group();
  Array1<Material*> matls;
  Array1<float> alphas;
  get_material(matls,alphas,do_phong);
  BrickArray3<Voxel> data;
  float data_min, data_max;
  Point minP, maxP;
  if (scene_type == 5) {
    // Do the nrrd stuff
    Nrrd *n = nrrdNew();
    // load the nrrd in
    if (nrrdLoad(n,nrrd_file)) {
      char *err = biffGet(NRRD);
      cerr << "Error reading nrrd "<< nrrd_file <<": "<<err<<"\n";
      free(err);
      biffDone(NRRD);
      return 0;
    }
    // check to make sure the dimensions are good
    if (n->dim != 3) {
      cerr << "VolumeVisMod error: nrrd->dim="<<n->dim<<"\n";
      cerr << "  Can only deal with 3-dimensional scalar fields... sorry.\n";
      return 0;
    }
    // convert the type to floats if you need to
    cerr << "Number of data members = " << n->num << endl;
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
      cerr << "Number of data members = " << n->num << endl;
    }
    // get the dimensions
    nx = n->axis[0].size;
    ny = n->axis[1].size;
    nz = n->axis[2].size;
    cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
    cout << "total = " << nz * ny * nz << endl;
    data.resize(nx,ny,nz); // resize the bricked data
    // get the physical bounds
    minP = Point(0,0,0);
    maxP = Point((nx - 1) * n->axis[0].spacing,
		 (ny - 1) * n->axis[1].spacing,
		 (nz - 1) * n->axis[2].spacing);
    // copy the data into the brickArray
    cerr << "Number of data members = " << n->num << endl;
    float *p = (float*)n->data; // get the pointer to the raw data
    for (int z = 0; z < nz; z++)
      for (int y = 0; y < ny; y++)
	for (int x = 0; x < nx; x++)
	  data(x,y,z).val = *p++;
    // compute the min and max of the data
    double dmin,dmax;
    nrrdMinMaxFind(&dmin,&dmax,n);
    data_min = (float)dmin;
    data_max = (float)dmax;
    // delete the memory that is no longer in use
    nrrdNuke(n);
  } else {
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
	    data(x,y,z).val = z;
	    break;
	  case 1:
	    data(x,y,z).val = y*z;
	    break;
	  case 2:
	    data(x,y,z).val = x*y*z;
	    break;
	  case 3:
	    data(x,y,z).val = y+z;
	    break;
	  case 4:
	    data(x,y,z).val = x+y+z;
	    break;
	  }
	}
      }
    }
    // compute the min and max of the data
    data_min = 0;
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
    }
    // set the physical dimensions of the data
    minP = Point(0,0,0);
    maxP = Point(nx,ny,nz);
    cout << "dim = (" << nx << ", " << ny << ", " << nz << ")\n";
    cout << "total = " << nz * ny * nz << endl;
  }
  
#if 0
  float data[2][3][4] = { { {0,1,2,3}, {0,1,2,3}, {0,1,2,3} },
			  { {0,1,2,3}, {0,1,2,3}, {0,1,2,3} } };

  //    { { 0, 1}, {0, 1}, {0, 1} },
  //			  { { 0, 1}, {0, 1}, {0, 1} },
  //			  { { 0, 1}, {0, 1}, {0, 1} },
  //			  { { 0, 1}, {0, 1}, {0, 1} } };
#endif
#if 0
  float data[24] = { 0,1, 0,1, 0,1,
		     0,1, 0,1, 0,1,
		     0,1, 0,1, 0,1,
		     0,1, 0,1, 0,1 };
#endif
  Object* obj = (Object*) new VolumeVis(data, data_min, data_max,
					nx, ny, nz,
					minP, maxP,
					&matls[0], matls.size(),
					&alphas[0], alphas.size());
  
  if(cut){
    PlaneDpy* pd=new PlaneDpy(Vector(0,0,1), Point(0,0,0));
    obj=(Object*)new CutPlane(obj, pd);
    new Thread(pd, "Cutting plane display thread");
  }
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



