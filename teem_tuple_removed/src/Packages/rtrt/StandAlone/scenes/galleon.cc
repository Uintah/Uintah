#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <Packages/rtrt/Core/Tri.h>
#include <ctype.h>
#include <Packages/rtrt/Core/Box.h>

using namespace rtrt;
using namespace std;

Array1<Point> points;
Group *g;
Material *mat;

Material *woodmat;
Material *sailmat;

Point c(0,0,0);

inline void Get3d(char *buf,Point &p)
{
  double x,y,z;
  if (3 != sscanf(buf,"%lf %lf %lf",&x,&y,&z)) {
    cerr << "Woah - bad point 3d!\n";
  }
  p = Point(x,y,z);
}

inline void Get2d(char *buf,Point &p)
{
  double x,y;
  if (2 != sscanf(buf,"%lf %lf",&x,&y)) {
    cerr << "Whoah - bad point 2d!\n";
  }
  p = Point (x,y,0);
}

char * GetNum(char *str, int &num)
{
    //int base=1;
  int index=0;
  
  while (str[index] && (!isdigit(str[index]))) index++;

  if (!str[index])
    return 0;
  num=0;
  while(str[index] && isdigit(str[index])) {
    num *= 10; // shift it over
    num += str[index]-'0';
    index++;
  }

  num--; // make it 0 based...

  return &str[index];
}

void GetFace(char *buf)
{
  static Array1<int> fis;
  //static Array1<int> uvis;
  //static Array1<int> nrmis;

  fis.resize(0);

  char *wptr=buf;
  int val;
  int what=0; // fi's

  while(wptr = GetNum(wptr,val)) {
    switch(what) {
    case 0:
      fis.add(val);
      break;
    case 1:
      break;
    case 2:
      break;
    default:
      cerr << "to many objects in face list!\n";
      return;
    }
    if (wptr[0]) {
      if (wptr[0] == '/') {
	what++;
	if (wptr[1] == '/') {
	  what++;
	  wptr++; // skip if no uvs...
	}
      } else {
	what=0; // back to faces...
      }
      wptr++; // bump it along...
    }
  }

  for(int k=0;k<fis.size()-2;k++) {
    int s0=0;
    int s1=1 + k;
    int s2=2 + k;

    Point p1 = points[fis[s0]];
    Point p2 = points[fis[s1]];
    Point p3 = points[fis[s2]];
    // check for degenerate triangles
#if 0
    if (p1 == p2) {
      cerr << "Degenerate triangle caught(1==2)\n";
      continue;
    }
    if (p1 == p3 || p2 == p3) {
      cerr << "Degenerate triangle caught(1==3 || 2==3)\n";
      continue;
    }
    g->add(new Tri(mat, p1, p2, p3));
#else
    Tri * t = new Tri(mat, p1, p2, p3);
    if (!(t->isbad())) {
      g->add(t);
    } else {
      delete(t);
      cerr << "Degenerate triangle!\n";
    }
#endif
    //g->add(new Tri(mat,points[fis[s0]],points[fis[s1]],points[fis[s2]]));
  }

}

void
parseobj(char *fname) {

   FILE *f=fopen(fname,"r");
   Point scrtchP;
 
   if (!f) {
     cerr << fname << " Woah - bad file name...\n";
   }
   
   char buf[4096];
   while(fgets(buf,4096,f)) {
     switch(buf[0]) {
     case 'v': // see wich type of vertex...
       {
	 switch(buf[1]) {
	 case 't': // texture coordinate...
	   Get2d(&buf[2],scrtchP);
	   break;
	 case 'n': // normal
	   Get3d(&buf[2],scrtchP);
	   break;
	 case ' ': // normal vertex...
	 default:
	   Get3d(&buf[2],scrtchP);
	   // add to points list!
	   points.add(scrtchP);
	   c = c + scrtchP.vector();
	  break;
	 }
	 break;
       }
     case 'f': // see which type of face...
       // Add tri to g
       GetFace(&buf[2]);
       break;
     case 'g':
       if (strncmp(&buf[2],"sail",strlen("sail")) == 0) {
	 mat = sailmat;
       } else {
	 mat = woodmat;
       }
       break;
     }
   }
}

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }
  Point Eye(0,0,200);
  Point Lookat(0,0,0);
  Vector Up(0,1,0);
  double fov=60;
  Camera cam(Eye,Lookat,Up,fov);
  double bgscale=0.5;
  //Color groundcolor(.82, .62, .62);
  //Color averagelight(1,1,.8);
  double ambient_scale=.5;
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  //Material *glass = new DielectricMaterial(1.33333, 1.0);
  Material *glass = new DielectricMaterial(1.5, 1.0, 0.04, 100.0, Color(.85, .97, .9), Color(1,1,1));
  Material *glass2 = new DielectricMaterial(1.0, 1.5, 0.04, 100.0, Color(.85, .97, .9), Color(1,1,1));

  //Sphere *s;
  //Box *box;

  BBox b;
  Point pmin,pmax,c;
  double r;

  g = new Group();
  mat = new LambertianMaterial(Color(1,0,0));
  woodmat = new LambertianMaterial(Color(.61,.164,.164));
  sailmat = new LambertianMaterial(Color(1,1,1));

  points.resize(0);

  parseobj("/opt/SCIRun/data/Geometry/models/galleon.obj");

  g->compute_bounds(b,0);
  pmin = b.min();
  pmax = b.max();

  c = c/points.size();
  r = 60;

  g->add(new Sphere(glass,c,r));
  g->add(new Sphere(glass2,c,0.90*r));

  printf("Pmin: %lf %lf %lf\n",pmin.x(),pmin.y(),pmin.z());
  printf("Pmax: %lf %lf %lf\n",pmax.x(),pmax.y(),pmax.z());

  Color cup(0.82, 0.52, 0.22);
  Color cdown(0.03, 0.05, 0.35);
  
  rtrt::Plane groundplane ( Point(1000, 0, 0), Vector(0, 2, 1) );
  Scene *scene = new Scene(g,cam,bgcolor,cdown, cup,groundplane,ambient_scale,
			   Arc_Ambient);
  scene->set_background_ptr( new LinearBackground(
                                                  Color(1.0, 1.0, 1.0),
                                                  Color(0.0,0.0,0.0),
                                                  Vector(0,0,1)) );
  scene->select_shadow_mode( Hard_Shadows );
  scene->add_light(new Light(Point(5000,-3,3), Color(1,1,.8), 0));
  return scene;


}
