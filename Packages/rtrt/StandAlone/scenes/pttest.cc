// XXX - ImageMaterial --> SharedTexture

#include <iostream>
#include <math.h>
#include <string.h>

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/ImageMaterial.h>

using namespace rtrt;

extern "C" 
Scene* make_scene(int argc, char** argv, int /*nworkers*/)
{
  char *bgname="/home/sci/cgribble/research/datasets/mpm/misc/envmap.ppm";
  char *texfile0="/home/sci/cgribble/research/datasets/mpm/misc/dummy0.ppm";
  char *texfile1="/home/sci/cgribble/research/datasets/mpm/misc/dummy1.ppm";
  char *texfile2="/home/sci/cgribble/research/datasets/mpm/misc/dummy2.ppm";
  char *texfile3="/home/sci/cgribble/research/datasets/mpm/misc/dummy3.ppm";
  for (int i=1;i<argc;i++)
  {
    if (strcmp(argv[i],"-bg")==0)
      bgname=argv[++i];
    else if (strcmp(argv[i],"-tex0")==0)
      texfile0=argv[++i];
    else if (strcmp(argv[i],"-tex1")==0)
      texfile1=argv[++i];
    else if (strcmp(argv[i],"-tex2")==0)
      texfile2=argv[++i];
    else if (strcmp(argv[i],"-tex3")==0)
      texfile3=argv[++i];
    else
    {
      cerr << "unrecognized option \"" << argv[i] << "\"" << endl;
      exit(1);
    }
  }
  
  Group* group=new Group();

  ImageMaterial* matl0 = new ImageMaterial(texfile0,
    ImageMaterial::Clamp,
    ImageMaterial::Clamp, 1,
    Color(0,0,0), 0);
  if (!matl0->valid())
  {
    cerr << "invalid" << endl;
    return 0;
  }
  Object* obj0=new Sphere(matl0, Point(-1,-1,0), 1 );
  group->add( obj0 );

  ImageMaterial* matl1 = new ImageMaterial(texfile1,
    ImageMaterial::Clamp,
    ImageMaterial::Clamp, 1,
    Color(0,0,0), 0);
  if (!matl1->valid())
  {
    cerr << "invalid" << endl;
    return 0;
  }
  Object* obj1=new Sphere(matl1, Point(-1,1,0), 1 );
  group->add( obj1 );

  ImageMaterial* matl2 = new ImageMaterial(texfile2,
    ImageMaterial::Clamp,
    ImageMaterial::Clamp, 1,
    Color(0,0,0), 0);
  if (!matl2->valid())
  {
    cerr << "invalid" << endl;
    return 0;
  }
  Object* obj2=new Sphere(matl2, Point(1,1,0), 1 );
  group->add( obj2 );

  ImageMaterial* matl3 = new ImageMaterial(texfile3,
    ImageMaterial::Clamp,
    ImageMaterial::Clamp, 1,
    Color(0,0,0), 0);
  if (!matl3->valid())
  {
    cerr << "invalid" << endl;
    return 0;
  }
  Object* obj3=new Sphere(matl3, Point(1,-1,0), 1 );
  group->add( obj3 );

  Camera cam(Point(0,0,-10), Point(0,0,0), Vector(0,1,0), 45.0);

  double ambient_scale=1.0;
  Color bgcolor(0,0,0);
  Color cdown(1,1,1);
  Color cup(1,1,1);

  rtrt::Plane groundplane(Point(0,0,-0.5), Vector(0,0,1));
  Scene* scene=new Scene(group, cam, bgcolor, cdown, cup, groundplane,
    ambient_scale, Arc_Ambient);

  EnvironmentMapBackground *emap=new EnvironmentMapBackground(bgname,
      Vector(0,1,0));
  scene->set_background_ptr(emap);
    
  Light* mainLight = new Light(Point(1,1,-10), Color(1,1,1), 0.8, 1.0 );
  mainLight->name_ = "main light";
  scene->add_light( mainLight );
  
  return scene;
}
