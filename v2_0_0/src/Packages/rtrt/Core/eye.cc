#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Mesh.h>
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
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/BSplineMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>

using namespace rtrt;
using namespace std;

#define MAXBUFSIZE 256
#define SCALE 950

void rotate(double /*theta*/, double /*phi*/)
{
}

void add_Rectangle(Group * parent, ifstream & in, double /* gamma */)
{
  char texture_file[1000];
  Point p0;
  Vector u,v;
  
  // while there is stuff in the file
  while(in){
    char file[1000];
    // stick the next line in file
    in >> file;
    if (in) {
      if (strcmp(file,"<DATA>") == 0) {
	cerr << "-------------Reading Data----------\n";
	// need to grab p0, u, and v
	double x,y,z;
	in >> x >> y >> z;
	p0 = Point(x,y,z);
	in >> x >> y >> z;
	u = Vector(x,y,z);
	in >> x >> y >> z;
	v = Vector(x,y,z);
      }
      else if (strcmp(file,"</DATA>") == 0) {
	cerr << "-------------End of Reading Data----------\n";
      }
      else if (strcmp(file,"<FILENAME>") == 0) {
	cerr << "-------------Reading filename----------\n";
	// grab the filename
	in >> texture_file;
      }
      else if (strcmp(file,"</FILENAME>") == 0) {
	cerr << "-------------End of Reading Filename----------\n";
      }
      else if (strcmp(file,"</RECTANGLE>") == 0) {
	cerr << "-------------Finishing Rectangle----------\n";
	// allocate memory for the parallelogram and return
#if 0
	Material* texture = new BSplineMaterial(texture_file,
						BSplineMaterial::Clamp,
						BSplineMaterial::Clamp,
						Color(0,0,0), 1,
						Color(0,0,0), 0);
#else
	Material* texture = new ImageMaterial(texture_file,
					      ImageMaterial::Clamp,
					      ImageMaterial::Clamp,
					      Color(0,0,0), 1,
					      Color(0,0,0), 0);
#endif
	parent->add(new Parallelogram(texture,p0,u,v));
	return;
      }
      else {
	cerr << "Error:eye.mo:add_Rectangle: Unknown file format!\n";
	cerr << "Don't know how to parse: " << file << endl;
      }
    }
  }
  
}

void add_Group(Group * parent, ifstream & in, double gamma)
{
  Group* g = new Group();
  
  // while there is stuff in the file
  while(in){
    char file[1000];
    // stick the next line in file
    in >> file;
    if (in) {
      if (strcmp(file,"<GROUP>") == 0) {
	cerr << "-------------Starting group----------\n";
	add_Group(g,in,gamma);
      }
      else if (strcmp(file,"</GROUP>") == 0) {
	cerr << "-------------Ending group----------\n";
	parent->add(g);
	return;
      }
      else if (strcmp(file,"<RECTANGLE>") == 0) {
	cerr << "-------------Starting rectangle----------\n";
	add_Rectangle(g,in,gamma);
      }
      else {
	cerr << "Error:eye.mo:add_Group: Unknown file format!\n";
	cerr << "Don't know how to parse: " << file << endl;
      }
    }
  }
  
}


extern "C"
Scene* make_scene(int argc, char* argv[])
{
  Array2<float> a;
  a.resize(1000,1000);
  for(int x=0; x<1000; x++)
    for(int y=0; y<1000; y++)
      a(x,y) = x + y;
  cerr << "Hello there" << endl;
  flush(cerr);
  if (argc < 2) {
    cerr << "usage: eye [header file] [gamma correction]\n";
  }

  double gamma_correction = 1;
  if (argc > 2) {
    gamma_correction = atof(argv[2]);
  }
  
  // parse the header file
  ifstream in(argv[1]);
  Group* g = new Group();
  
  // while there is stuff in the file
  while(in){
    char file[1000];
    // stick the next line in file
    in >> file;
    if (in) {
      if (strcmp(file,"<GROUP>") == 0) {
	cerr << "-------------Starting group----------\n";
	add_Group(g,in,gamma_correction);
      }
      else {
	cerr << "Error:eye.mo:main Unknown file format!\n";
	cerr << "Don't know how to parse: " << file << endl;
      }
    }
  }
  


  //Point Eye(800,1590,360);
  Point Eye(300,300,-1000);
  Point Lookat(300,300,1500);
  Vector Up(0,1,0);
  double fov=45;
  
  Camera cam(Eye,Lookat,Up,fov);

  double bgscale=1;
  Color groundcolor(0,0,0);
  Color averagelight(1,1,1);
  double ambient_scale=1;
  
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  
  Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
  Scene* scene=new Scene(g, cam, bgcolor, groundcolor*averagelight,
			 bgcolor, groundplane, ambient_scale);
  //  scene->add_light(new Light(Point(5,-3,3), Color(1,1,.8)*2, 0));
  scene->set_background_ptr( new LinearBackground(
						  Color(0.2, 0.4, 0.9),
						  Color(0.4,0.8,1),
						  Vector(0, 1, 0)) );
  //Color bgcolor(0.8, 0.5, 0.5);
  //Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
  //Color cdown(0.1, 0.1, 0.7);
  //Color cup(0.8, 0.8, 0.4);
  //double ambient_scale=.5;
  //Scene *scene = new Scene(g,cam,bgcolor,cdown, cup,groundplane,ambient_scale);
  
  //scene->shadow_mode = 1;
  //scene->maxdepth = 8;
  //  scene->shadowobj = new BV1(shadow);
  //scene->add_light(new Light(Point(200,400,1300), Color(.8,.8,.8), 0));
  //scene->animate=false;
  return scene;
}



