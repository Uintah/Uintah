

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/UVSphere.h>
#include <Packages/rtrt/Core/SharedTexture.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <math.h>
#include <string.h>

using namespace rtrt;

Group *make_geometry(char* tex0, char* tex1,
		     char* tex2, char* tex3)
{
  Group* group=new Group();
  
  SharedTexture* matl0 = new SharedTexture(tex0);
  if (!matl0->valid())
  {
    cerr << "matl0 is invalid" << endl;
    return 0;
  }
  Object* obj0=new UVSphere(matl0, Point(0,0,0), 1, Vector(0,1,0));
  group->add( obj0 );
  
  /*
  SharedTexture* matl0 = new SharedTexture(tex0);
  if (!matl0->valid())
  {
    cerr << "matl0 is invalid" << endl;
    return 0;
  }
  Object* obj0=new UVSphere(matl0, Point(-1,-1,0), 1, Vector(0,1,0));
  group->add( obj0 );

  SharedTexture* matl1 = new SharedTexture(tex1);
  if (!matl1->valid())
  {
    cerr << "matl1 is invalid" << endl;
    return 0;
  }
  Object* obj1=new UVSphere(matl1, Point(-1,1,0), 1, Vector(0,1,0));
  group->add( obj1 );

  SharedTexture* matl2 = new SharedTexture(tex2);
  if (!matl2->valid())
  {
    cerr << "matl2 is invalid" << endl;
    return 0;
  }
  Object* obj2=new UVSphere(matl2, Point(1,1,0), 1, Vector(0,1,0));
  group->add( obj2 );

  SharedTexture* matl3 = new SharedTexture(tex3);
  if (!matl3->valid())
  {
    cerr << "matl3 is invalid" << endl;
    return 0;
  }
  Object* obj3=new UVSphere(matl3, Point(1,-1,0), 1, Vector(0,1,0));
  group->add( obj3 );
  */
  
  return group;
}

extern "C" 
Scene* make_scene(int argc, char** argv, int /*nworkers*/)
{
  char *bg="/home/sci/cgribble/research/datasets/mpm/misc/envmap.ppm";
  char *tex0="/home/sci/cgribble/SCIRun/irix.64/Packages/rtrt/StandAlone/sphere0.ppm";
  char *tex1="/home/sci/cgribble/SCIRun/irix.64/Packages/rtrt/StandAlone/sphere1.ppm";
  char *tex2="/home/sci/cgribble/SCIRun/irix.64/Packages/rtrt/StandAlone/sphere2.ppm";
  char *tex3="/home/sci/cgribble/SCIRun/irix.64/Packages/rtrt/StandAlone/sphere3.ppm";
  for (int i=1;i<argc;i++)
  {
    if (strcmp(argv[i],"-bg")==0)
      bg=argv[++i];
    else if (strcmp(argv[i],"-tex0")==0)
      tex0=argv[++i];
    else if (strcmp(argv[i],"-tex1")==0)
      tex1=argv[++i];
    else if (strcmp(argv[i],"-tex2")==0)
      tex2=argv[++i];
    else if (strcmp(argv[i],"-tex3")==0)
      tex3=argv[++i];
    else
    {
      cerr << "unrecognized option \"" << argv[i] << "\"" << endl;
      exit(1);
    }
  }
 
  Group *group=make_geometry(tex0, tex1, tex2, tex3);

  Camera cam(Point(0,0,10), Point(0,0,0), Vector(0,1,0), 45.0);

  double ambient_scale=1.0;
  Color bgcolor(0,0,0);
  Color cdown(1,1,1);
  Color cup(1,1,1);

  rtrt::Plane groundplane(Point(0,0,0), Vector(0,1,0));
  Scene* scene=new Scene(group, cam, bgcolor, cdown, cup, groundplane,
    ambient_scale, Arc_Ambient);

  EnvironmentMapBackground *emap=new EnvironmentMapBackground(bg, Vector(0,1,0));
  scene->set_background_ptr(emap);
    
  Light* mainLight = new Light(Point(10,10,10), Color(1,1,1), 1.0);
  mainLight->name_ = "main light";
  scene->add_light( mainLight );
  
  return scene;
}
