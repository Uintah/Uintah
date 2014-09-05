#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/Point.h>
#include <Packages/rtrt/Core/Vector.h>
#include <Packages/rtrt/Core/Mesh.h>
#include <Packages/rtrt/Core/Bezier.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/MetalMaterial.h>

using namespace rtrt;

#define MAXBUFSIZE 256

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
      for(int i=1;i<argc;i++) {
	cerr << "Unknown option: " << argv[i] << '\n';
	cerr << "Valid options for scene: " << argv[0] << '\n';
	return 0;
      }

      Point Eye(10,235,130);
      Point Lookat(0,0,0);
      Vector Up(0,0,1);
      double fov=60;

      //Point Eye(0,0,2);
      //Point Lookat(0,0,0);
      //Vector Up(0,1,0);
      //double fov = 90;

      Camera cam(Eye,Lookat,Up,fov);

      double bgscale=0.5;
      //Color groundcolor(.82, .62, .62);
      //Color averagelight(1,1,.8);
      double ambient_scale=.5;
      int subdivlevel = 3;
      

      Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

      //Material* red_shiny = new Phong(Color(0,0,0), Color(1,0,0), Color(1,1,1), 25, 0);
      //Material* white = new LambertianMaterial(Color(1,1,1));
      Material* silver = new MetalMaterial(Color(0.7,0.73,0.8));

      char buf[MAXBUFSIZE];
      char *name;
      Group *g = new Group();
      FILE *fp;
      double x,y,z;
      
      fp = fopen("teapot.dat","r");

      while (fscanf(fp,"%s",buf) != EOF) {
	if (!strcasecmp(buf,"bezier")) {
	  int numumesh, numvmesh, numcoords=3;
	  Mesh *m;
	  Bezier *b;
	  Point p;

	  fscanf(fp,"%s",buf);
	  name = new char[strlen(buf)+1];
	  strcpy(name,buf);
	  
	  fscanf(fp,"%d %d %d\n",&numumesh,&numvmesh,&numcoords);
	  m = new Mesh(numumesh,numvmesh);
	  for (int j=0; j<numumesh; j++) {
	    for (int k=0; k<numvmesh; k++) {
	      fscanf(fp,"%lf %lf %lf",&x,&y,&z);
	      p = Point(x,y,z);
	      m->mesh[j][k] = p;
	    }
	  }
	  b = new Bezier(silver,m);
	  b->SubDivide(subdivlevel,.5);
	  g->add(b->MakeBVH());
        }
      }
      fclose(fp);
      
      Color cup(0.82, 0.52, 0.22);
      Color cdown(0.03, 0.05, 0.35);

      Plane groundplane ( Point(1000, 0, 0), Vector(0, 2, 1) );
      Scene *scene = new Scene(g,cam,bgcolor,cdown, cup,groundplane,ambient_scale);
      scene->ambient_hack = true;
      scene->set_background_ptr( new LinearBackground(
                               Color(1.0, 1.0, 1.0),
                               Color(0.0,0.0,0.0),
                               Vector(0,0,1)) );



      scene->shadow_mode = 1;
      scene->add_light(new Light(Point(5000,-3,3), Color(1,1,.8), 0));
      return scene;
}



