#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/GridTris.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <Packages/rtrt/Core/ply.h>
#include <Packages/rtrt/Core/PLYReader.h>

using namespace rtrt;

extern "C" 
Scene* make_scene(int argc, char* argv[])
{
//    for(int i=1;i<argc;i++) {
//      cerr << "Unknown option: " << argv[i] << '\n';
//      cerr << "Valid options for scene: " << argv[0] << '\n';
//      return 0;
//    }
// -eye -1329.7 333.468 -8572.81 -lookat 79.9753 34.1933 -13346.8 -up -0.168284 0.979459 -0.111091 -fov 60
  char* file = "/opt/SCIRun/data/Geometry/Stanford_Sculptures/david_1mm.ply";
  //char* file = "/opt/SCIRun/data/Geometry/Stanford_Sculptures/happy_vrip_res2.ply";
  int cells=5;
  int depth=3;
  double scale = .001;
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-cells") == 0){
      i++;
      cells = atoi(argv[i]);
    } else if(strcmp(argv[i], "-depth") == 0){
      i++;
      depth = atoi(argv[i]);
    } else if(strcmp(argv[i], "-file") == 0){
      i++;
      file = argv[i];
    } else if(strcmp(argv[i], "-scale") == 0){
      i++;
      scale = atof(argv[i]);
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      exit(1);
    }
  }

//  Point Eye(-1329.7, 333.468, -8572.81);
//  Point Lookat(79.9753, 34.1933, -13346.8);
//  Vector Up(-0.168284, 0.979459, -0.111091);
//  double fov=60;

  Point Eye(-17.1721, -34.6021, 3.4593);
  Point Lookat(-14, -20, 3.40703);
  Vector Up(0, 0, 1);
  double fov=22.5;

  Camera cam(Eye,Lookat,Up,fov);

  Color bone(0.9608, 0.8706, 0.7020);
  Material* matl0=new Phong(bone*.6, bone*.6, 100, 0);
  //Material *flat_white = new LambertianMaterial(Color(0,0,1));

  //GridTris* david = new GridTris(flat_white, 100, 0);
  string gridfile(file);
  gridfile = string(gridfile, 0, gridfile.size()-4) + "-grid";
  cerr << "gridfile = ("<<gridfile<<")\n";
  GridTris* david = new GridTris(matl0, cells, depth, gridfile);
  Point dav_ped_top(-14,-20,1);

  cerr << "reading ply\n";
  read_ply(file,david);
  cerr << "done reading ply\n";
  BBox david_bbox;

  david->compute_bounds(david_bbox,0);

  Point min = david_bbox.min();
  Point max = david_bbox.max();
  Vector diag = david_bbox.diagonal();
  printf("BBox: min: %lf %lf %lf max: %lf %lf %lf\nDimensions: %lf %lf %lf\n",
	 min.x(),min.y(), min.z(),
	 max.x(),max.y(), max.z(),
	 diag.x(),diag.y(),diag.z());
  Transform davidT;

  davidT.pre_translate(-Vector((max.x()+min.x())/2.,(max.y()+min.y())/2.,min.z())); // center david over 0
  davidT.pre_rotate(M_PI,Vector(0,0,1));  // Face front
  davidT.pre_scale(Vector(scale,scale,scale)); // make units meters
  davidT.pre_translate(dav_ped_top.asVector());

  davidT.print();

  david->transform(davidT);
  david_bbox.reset();
  david->compute_bounds(david_bbox,0);

  min = david_bbox.min();
  max = david_bbox.max();
  diag = david_bbox.diagonal();

  printf("BBox: min: %lf %lf %lf max: %lf %lf %lf\nDimensions: %lf %lf %lf\n",
	 min.x(),min.y(), min.z(),
	 max.x(),max.y(), max.z(),
	 diag.x(),diag.y(),diag.z());

  double bgscale=0.5;
  //Color groundcolor(.82, .62, .62);
  //Color averagelight(1,1,.8);
  double ambient_scale=.5;
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  Color cup(0.82, 0.52, 0.22);
  Color cdown(0.03, 0.05, 0.35);
  
  Plane groundplane ( Point(1000, 0, 0), Vector(0, 2, 1) );
  Scene *scene = new Scene(david,cam,bgcolor,cdown, cup,groundplane,ambient_scale, Arc_Ambient);
  scene->set_background_ptr( new LinearBackground(
						  Color(1.0, 1.0, 1.0),
						  Color(0.1,0.1,0.1),
						  Vector(0,0,1)) );
  
  scene->select_shadow_mode(Hard_Shadows);

  Light *david_light = new Light(Point(-10,-50,20), Color(1,1,0.9), 1);
  david_light->name_ = "david_light";
  scene->add_light(david_light);

  Light *david_light_back = new Light(Point(3,10,10), Color(1,1,1), 1);
  david_light_back->name_ = "david_light_back";
  scene->add_light(david_light_back);

  return scene;
}
