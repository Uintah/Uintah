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
  char* file = "/opt/SCIRun/data/Geometry/Stanford_Sculptures/david_head_1mm_color.ply";
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

  Point Eye(-12.4558, -10.5506, 1.6543);
  Point Lookat(-10.0211, -11.0325, 1.46267);
  Vector Up(0.0897179, 0.132826, 0.9870);
  double fov=30.0;

  Camera cam(Eye,Lookat,Up,fov);

  Color bone(0.9608, 0.8706, 0.7020);
  Material* matl0=new Phong(bone*.6, bone*.6, 100, 0);
  //Material *flat_white = new LambertianMaterial(Color(0,0,1));

  //GridTris* david = new GridTris(flat_white, 100, 0);
  char newfile[1000];
  strncpy(newfile, file, strlen(file)-4);
  string newfileS = string(newfile)+"-grid";
  GridTris* david = new GridTris(matl0, cells, depth,
                                 newfileS);
  Point dav_ped_top(-14,-20,1);

  read_ply(file,david);
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

  davidT.pre_translate(-Vector((max.x()+min.x())/2.,min.y(),(max.z()+min.z())/2.)); // center david over 0
  davidT.pre_rotate(M_PI_2,Vector(1,0,0));  // make z up
  davidT.pre_scale(Vector(scale,scale,scale)); // make units meters
  davidT.pre_translate(dav_ped_top.asVector());

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
  Scene *scene = new Scene(david,cam,bgcolor,cdown, cup,groundplane,ambient_scale);
  scene->set_background_ptr( new LinearBackground(
						  Color(1.0, 1.0, 1.0),
						  Color(0.1,0.1,0.1),
						  Vector(0,0,1)) );
  
  scene->select_shadow_mode(Hard_Shadows);
  Light *david_light = new Light(Point(-50,30,20), Color(1,1,1), 0);

  david_light->name_ = "david_light";

  scene->add_light(david_light);
  return scene;

}
