

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/UVSphere2.h>
#include <Packages/rtrt/Core/SharedTexture.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <math.h>
#include <string.h>

using namespace rtrt;

#define NUM_TEXTURES 8

float radius = 1.0;

/*
  -eye -4.05479 0.0260826 7.411 -lookat 0 0 0 -up 0.125951 0.989877 0.065428 -fov 42.5676

*/

// Returns 1 if there was an error.  This is based on if the texture
// was found.
int add_sphere(char *tex_name, const Point& center, Group *group) {
  SharedTexture* matl = new SharedTexture(tex_name);
  if (!matl->valid())
  {
    cerr << "AddSphere::texture is bad :" << tex_name << endl;
    return 1;
  }
  group->add( new UVSphere2(matl, center, radius) );
  return 0;
}

Group *make_geometry(char* tex_names[NUM_TEXTURES])
{
  Group* group=new Group();

  int E = 0;
  if (!E) E |= add_sphere(tex_names[0], Point(-1,-1,-1), group);
  if (!E) E |= add_sphere(tex_names[1], Point(-1, 1,-1), group);
  if (!E) E |= add_sphere(tex_names[2], Point( 1, 1,-1), group);
  if (!E) E |= add_sphere(tex_names[3], Point( 1,-1,-1), group);

  if (!E) E |= add_sphere(tex_names[4], Point(-1,-1, 1), group);
  if (!E) E |= add_sphere(tex_names[5], Point(-1, 1, 1), group);
  if (!E) E |= add_sphere(tex_names[6], Point( 1, 1, 1), group);
  if (!E) E |= add_sphere(tex_names[7], Point( 1,-1, 1), group);
  
  if (!E)
    return group;
  else
    return 0;
}

extern "C" 
Scene* make_scene(int argc, char** argv, int /*nworkers*/)
{
  char *bg="/home/sci/cgribble/research/datasets/mpm/misc/envmap.ppm";
  char *tex_basename="./sphere";

  for (int i=1;i<argc;i++)
  {
    if (strcmp(argv[i],"-bg")==0)
      bg = argv[++i];
    else if (strcmp(argv[i],"-tex")==0)
      tex_basename = argv[++i];
    else if(strcmp(argv[i],"-radius")==0)
      radius=atof(argv[++i]);
    else
    {
      cerr << "unrecognized option \"" << argv[i] << "\"" << endl;
      exit(1);
    }
  }

  char *tex_names[NUM_TEXTURES];
  // Make the tex_names
  size_t name_length = strlen(tex_basename) + 15;
  for(int i = 0; i < NUM_TEXTURES; i++) {
    tex_names[i] = new char[name_length];
    sprintf(tex_names[i], "%s%d.ppm", tex_basename, i); 
  }
  
  Group *group=make_geometry(tex_names);
  if (!group)
    // Then something went wrong and you should kill the scene
    return 0;

  Camera cam(Point(0,0,10), Point(0,0,0), Vector(0,1,0), 45.0);

  double ambient_scale=2;
  Color bgcolor(0,0,0);
  Color cdown(1,1,1);
  Color cup(1,1,1);

  rtrt::Plane groundplane(Point(0,0,0), Vector(0,1,0));
  Scene* scene=new Scene(group, cam, bgcolor, cdown, cup, groundplane,
    ambient_scale, Arc_Ambient);

  EnvironmentMapBackground *emap=new EnvironmentMapBackground(bg, Vector(0,1,0));
  if (emap->valid() != true) {
    // try a local copy
    delete emap;
    emap = new EnvironmentMapBackground("./envmap.ppm", Vector(0,1,0));
    if (emap->valid() != true) {
      return 0;
    }
  }
  scene->set_background_ptr(emap);
    
  Light* mainLight = new Light(Point(-5,10,7.5), Color(1,1,1), 0.01);
  mainLight->name_ = "main light";
  scene->add_light( mainLight );
  scene->turnOffAllLights( 0.0 ); 

  scene->select_shadow_mode(No_Shadows);
  
  return scene;
}
