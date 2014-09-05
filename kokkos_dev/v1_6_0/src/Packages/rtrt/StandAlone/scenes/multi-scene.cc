#include <Packages/rtrt/Core/BouncingSphere.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <dlfcn.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <sys/param.h>

using namespace rtrt;
using namespace std;

// -scene scenes/multiscene [nscenes] -scene scenes/xxx arg0 ... argn -scene scenes/yyy arg0 ... argm

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  int nscenes = atoi(argv[1]);
  cerr << "multi-scene will read "<<nscenes<<" scenes (argc="<<argc<<").\n";
  Array1<Scene *> scene(nscenes);
  Array1<int> scene_argc(nscenes);
  Array1<int> scene_first_arg(nscenes);
  int curr_arg=2;
  int s;
  for (s=-1; s<nscenes && curr_arg<argc; curr_arg++) {
    if (strcmp(argv[curr_arg], "-scene") == 0) {
      s++;
      if (curr_arg+2>argc) {
	cerr << "Error - need a name for this scene!\n";
      }
      if (s!=0) scene_argc[s-1]=curr_arg-scene_first_arg[s-1];
      scene_first_arg[s]=curr_arg+1;
    }
    if (s<0) {
      cerr << "Error - didn't see '-scene' as first argument.\n";
      exit(1);
    }
  }
  if (curr_arg != argc) {
    cerr << "Error - found too many -scene arguments!\n";
    exit(1);
  }
  if (s != nscenes-1) {
    cerr << "Error - didn't find enough scenes!\n";
    exit(1);
  }
  scene_argc[s]=argc-scene_first_arg[s];
  
  for (s=0; s<nscenes; s++) {
    cerr << "Scene["<<s<<"] nargs="<<scene_argc[s]<<" args: ";
    for (int i=0; i<scene_argc[s]; i++) {
      cerr << argv[scene_first_arg[s]+i]<<" ";
    }
    cerr <<"\n";
  }

  Group *g = new Group;
  for (s=0; s<nscenes; s++) {
    char scenefile[MAXPATHLEN];
    sprintf(scenefile, "./%s.mo", argv[scene_first_arg[s]]);
    void* handle=dlopen(scenefile, RTLD_NOW);
    if(!handle){
      cerr << "Error opening scene: " << scene_first_arg[s] << '\n';
      cerr << dlerror() << '\n';
      exit(1);
    }
    void* scene_fn=dlsym(handle, "make_scene");
    if(!scene_fn){
      cerr << "Scene file found, but make_scene() function not found\n";
      exit(1);
    }
    Scene* (*make_scene)(int,char**,int) = (Scene*(*)(int,char**,int))scene_fn;
    scene[s]=(*make_scene)(scene_argc[s], &(argv[scene_first_arg[s]]), nworkers);
    if(!scene[s]){
      cerr << "Scene creation failed!\n";
      exit(1);
    }
    g->objs.add(scene[s]->get_object());
    if (s!=0)
      for (int l=0; l<scene[s]->nlights(); l++)
	scene[0]->add_light(scene[s]->light(l));
  }
  scene[0]->set_object(g);
  return scene[0];
}
