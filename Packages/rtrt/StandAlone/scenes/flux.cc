#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Group.h>
#ifdef USE_BRICK
#include <Packages/rtrt/Core/HVolumeBrick.h>
#else
#include <Packages/rtrt/Core/HVolume.h>
#include <Packages/rtrt/Core/BrickArray3.h>
#endif
#include <Packages/rtrt/Core/HVolumeBrickColor.h>
#include <Packages/rtrt/Core/HVolumeMaterial.h>
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
#include <Packages/rtrt/Core/Array1.cc>
#include <Packages/rtrt/Core/TimeObj.h>
#include <Core/Thread/Thread.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdlib.h>

using namespace rtrt;
using SCIRun::Thread;

#ifdef USE_BRICK
void create_grid(HVolumeBrick* hvol, TimeObj *group);
#else
void create_grid(HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > * hvol, TimeObj *group);
#endif

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  int depth=3;
  char* texfile=0;
  bool showgrid=false;
  bool xyslice=false;
  bool xzslice=false;
  bool yzslice=false;
  Array1<char*> files;
  bool cut=false;
  double rate=3;
  int num_non_trans = 1;
  int num_trans = 0;
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-depth")==0){
      i++;
      depth=atoi(argv[i]);
    } else if(strcmp(argv[i], "-showgrid")==0){
      showgrid=true;
    } else if(strcmp(argv[i], "-texture")==0){
      i++;
      texfile=argv[i];
    } else if(strcmp(argv[i], "-xyslice")==0){
      xyslice=true;
    } else if(strcmp(argv[i], "-xzslice")==0){
      xzslice=true;
    } else if(strcmp(argv[i], "-yzslice")==0){
      yzslice=true;
    } else if(strcmp(argv[i], "-cut")==0){
      cut=true;
    } else if(strcmp(argv[i], "-rate")==0){
      rate = atof(argv[++i]);
    } else if(strcmp(argv[i], "-nontrans")==0){
      num_non_trans = atoi(argv[++i]);
    } else if(strcmp(argv[i], "-trans")==0){
      num_trans = atoi(argv[++i]);
    } else if(argv[i][0] != '-'){
      files.add(argv[i]);
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      cerr << "Valid options for scene: " << argv[0] << '\n';
      cerr << " -depth n   - set depth of hierarchy\n";
      cerr << " file       - raw file name\n";
      return 0;
    }
  }

  if(files.size()==0){
    cerr << "Must specify at least one file\n";
    return 0;
  }
  Array1<Material*> matls;
  if(texfile){
    matls.add(new HVolumeBrickColor(texfile, nworkers,
				    .6, .7, .6, 50,  0));
  } else {
    CatmullRomSpline<Color> spline(0);
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
    int ncolors=5000;
    matls.resize(ncolors);
    float Ka=.8;
    float Kd=.8;
    float Ks=.8;
    float refl=0;
    float specpow=40;
    for(int i=0;i<ncolors;i++){
      float frac=float(i)/(ncolors-1);
      Color c(spline(frac));
      matls[i]=new Phong(c*Ka, c*Kd, c*Ks, specpow, refl);
      //matls[i]=new LambertianMaterial(c*Kd);
    }
  }
  Array1<Material*> trans_matls;
  if(texfile){
    trans_matls.add(new HVolumeBrickColor(texfile, nworkers,
				    .6, .7, .6, 50,  0));
  } else {
    CatmullRomSpline<Color> spline(0);
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
    int ncolors=5000;
    trans_matls.resize(ncolors);
    //    float Ka=.8;
    //float Kd=.8;
    //float Ks=.8;
    float refl=0;
    float specpow=40;
    for(int i=0;i<ncolors;i++){
      float frac=float(i)/(ncolors-1);
      Color c(spline(frac));
      trans_matls[i]=new PhongMaterial(c, 0.1, refl, specpow);
      //      trans_matls[i]=new Phong(c*Ka, c*Kd, c*Ks, specpow, refl);
      //matls[i]=new LambertianMaterial(c*Kd);
    }
  }

  // set up variables for the transparent materials
  Array1<HVolumeTransferFunct*> transfer(num_non_trans);
  Array1<VolumeDpy*> dpys(num_non_trans);
  for (int n = 0; n < num_non_trans; n++) {
    transfer[n] = new HVolumeTransferFunct(&matls[0], matls.size());
    dpys[n]=new VolumeDpy(0.5);
  }

  // set up variables for the non transparent materials
  Array1<HVolumeTransferFunct*> transfer_trans(num_trans);
  Array1<VolumeDpy*> dpys_trans(num_trans);
  for (int n = 0; n < num_trans; n++) {
    transfer_trans[n] = new HVolumeTransferFunct(&trans_matls[0],
						 trans_matls.size());
    dpys_trans[n]=new VolumeDpy(0.5);
  }
  
  Object* obj;
  TimeObj* group = new TimeObj(rate);
  obj=group;
  for(int i=0;i<files.size();i++){
    // try to open the temperature file
    char temp_file[500];
    sprintf(temp_file,"%s.temp",files[i]);
    ifstream temp_in(temp_file);
    Array1<Material*> hvol_matl(num_non_trans);
    Array1<Material*> hvol_matl_trans(num_trans);
    if(temp_in){
      // extract the data from the temperature file
      // number of rho's
      float min_rho,max_rho;
      temp_in >> min_rho >> max_rho;
      cout << "min_rho = "<<min_rho<<", max_rho = "<<max_rho<<endl;
      int num;
      temp_in >> num;
      Array1<float> data(num);
      unsigned int datai;
      for(datai = 0; datai < num; datai++)
	temp_in >> data[datai];
      cout << "num = "<< num<<", datai = "<<datai<<endl;
      // opaque materials
      for (int n = 0; n < num_non_trans; n++) {
	hvol_matl[n] = (Material*)new HVolumeMaterial(dpys[n], data, min_rho,
						   max_rho, transfer[n]);
      }
      // transparent materials
      for (int n = 0; n < num_trans; n++) {
	hvol_matl_trans[n] =
	  (Material*)new HVolumeMaterial(dpys_trans[n], data, min_rho,
					 max_rho, transfer_trans[n]);
      }
    } else {
      cerr << "Temperature file( "<< temp_file << " not found\n";
      for (int n = 0; n < num_non_trans; n++)
	hvol_matl[n] = matls[0];
      for (int n = 0; n < num_trans; n++)
	hvol_matl_trans[n] = trans_matls[0];
    }
    Group *timestep = new Group();
#ifdef USE_BRICK
    HVolumeBrick* hvol=new HVolumeBrick(hvol_matl, dpys[0], files[i],
					depth, nworkers);
#else
    HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > * hvol = new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > (hvol_matl[0], dpys[0], files[i], depth, nworkers);
    timestep->add(hvol);
    for (int n = 1; n < num_non_trans; n++) {
      hvol = new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > (hvol_matl[n], dpys[n], files[i], depth, nworkers);
      timestep->add(hvol);      
    }
    for (int n = 0; n < num_trans; n++) {
      hvol = new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > (hvol_matl_trans[n], dpys_trans[n], hvol);
      timestep->add(hvol);
    }
#endif
    group->add(timestep);
    if(showgrid){
      create_grid(hvol, group);
    }
  }

  // compute_min_max  should be called after all the HVolumeMaterial's
  // have been created and added
  // Start up the thread to handle the slider
  for (int n = 0; n < num_non_trans; n++) {
    transfer[n]->compute_min_max();
    new Thread(dpys[n], "Volume GUI thread");
  }
  for (int n = 0; n < num_trans; n++) {
    transfer_trans[n]->compute_min_max();
    new Thread(dpys_trans[n], "Volume GUI thread2");
  }
	
  
  if(xyslice || xzslice || yzslice){
    Group* group=new Group();
    group->add(obj);
    BBox bbox;
    obj->compute_bounds(bbox, 0);
    Vector diag(bbox.diagonal()*0.5);
    Point mid(bbox.min()+diag);
    if(xyslice){
      group->add(new Rect(matls[0], mid, Vector(diag.x(), 0, 0),
			  Vector(0, diag.y(), 0)));
    }
    if(xzslice){
      group->add(new Rect(matls[0], mid, Vector(diag.x(), 0, 0),
			  Vector(0, 0, diag.z())));
    }
    if(yzslice){
      group->add(new Rect(matls[0], mid, Vector(0, diag.y(), 0),
			  Vector(0, 0, diag.z())));
    }
    obj=group;
  }

  if(cut){
    PlaneDpy* pd=new PlaneDpy(Vector(0,0,1), Point(0,0,100));
    obj=new CutPlane(obj, pd);
    new Thread(pd, "Cutting plane display thread");
  }

  //double bgscale=0.5;
  double ambient_scale=.5;

  Color bgcolor(0.01, 0.05, 0.3);
  Color cup(1, 0, 0);
  Color cdown(0, 0, 0.2);

  Plane groundplane ( Point(0, 0, 0), Vector(0, -1, 0) );
  Camera cam(Point(1501.35, -482.14, -257.168),
	     Point(1461.09, 56.3614, 31.5762),
	     Vector(-1,0,0),
	     34.62);
  Scene* scene=new Scene(obj, cam,
			 bgcolor, cdown, cup, groundplane,
			 ambient_scale);
  //scene->add_light(new Light(Point(50,-30,30), Color(1.0,0.8,0.2), 0));
  scene->add_light(new Light(Point(1100,-600,3000), Color(1.0,1.0,1.0), 0));
  scene->set_background_ptr( new LinearBackground(
						  Color(0.2, 0.4, 0.9),
						  Color(0.0,0.0,0.0),
						  Vector(1, 0, 0)) );

  scene->shadow_mode=0;
  scene->ambient_hack=false;
  return scene;
}

#ifdef USE_BRICK
void create_grid(HVolumeBrick* hvol, TimeObj *group) {
  int nx=hvol->get_nx();
  int ny=hvol->get_ny();
  int nz=hvol->get_nz();
#else
void create_grid(HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > * hvol, TimeObj *group) {
  int nx=hvol->nx;
  int ny=hvol->ny;
  int nz=hvol->nz;
#endif
  Material* cylmatl=new LambertianMaterial( Color(0.3,0.3,0.3) );
  BBox bbox;
  hvol->compute_bounds(bbox, 0);
  Point min(bbox.min());
  Point max(bbox.max());
  Vector diag(max-min);
  double radius=Min(diag.x()/nx, diag.y()/ny, diag.z()/nz);
  radius/=16;
  for(int x=0;x<nx;x++){
    for(int y=0;y<ny;y++){
      double xn=double(x)/double(nx-1)*diag.x()+min.x();
      double yn=double(y)/double(ny-1)*diag.y()+min.y();
      group->add(new Cylinder(cylmatl,
			      Point(xn,yn,min.z()),
			      Point(xn,yn,max.z()),
			      radius));
    }
  }
  for(int x=0;x<nx;x++){
    for(int z=0;z<nz;z++){
      double xn=double(x)/double(nx-1)*diag.x()+min.x();
      double zn=double(z)/double(nz-1)*diag.z()+min.z();
      group->add(new Cylinder(cylmatl,
			      Point(xn,min.y(),zn),
			      Point(xn,max.z(),zn),
			      radius));
    }
  }
  for(int z=0;z<nz;z++){
    for(int y=0;y<ny;y++){
      double zn=double(z)/double(nz-1)*diag.z()+min.z();
      double yn=double(y)/double(nz-1)*diag.y()+min.y();
      group->add(new Cylinder(cylmatl,
			      Point(min.z(),yn,zn),
			      Point(max.z(),yn,zn),
			      radius));
    }
  }
  for(int x=0;x<nx;x++){
    for(int y=0;y<ny;y++){
      for(int z=0;z<nz;z++){
	double xn=double(x)/double(nx-1)*diag.x()+min.x();
	double yn=double(y)/double(nz-1)*diag.y()+min.y();
	double zn=double(z)/double(nz-1)*diag.z()+min.z();
	group->add(new Sphere(cylmatl, Point(xn,yn,zn), radius));
      }
    }
  }
}


  
