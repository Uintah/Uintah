#include <Packages/rtrt/Core/BouncingSphere.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/CellGroup.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#if !defined(linux)
#  include <Packages/rtrt/Sound/Sound.h>
#  include <Packages/rtrt/Core/Trigger.h>
#endif

#include <dlfcn.h>
#include <math.h>
#include <string.h>

#include <iostream>
#include <vector>

#include <sys/param.h>

using namespace rtrt;
using namespace std;

// -scene scenes/multiscene [nscenes] -scene scenes/xxx arg0 ... argn -scene scenes/yyy arg0 ... argm

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  int nscenes = atoi(argv[1]);
  cerr << "multi-scene will read "<<nscenes<<" scenes (argc="<<argc<<").\n";
  rtrt::Array1<Scene *> scene(nscenes);
  rtrt::Array1<int> scene_argc(nscenes);
  rtrt::Array1<int> scene_first_arg(nscenes);
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

  CellGroup *g = new CellGroup;

#if !defined(linux)
  vector<Trigger*>          * allTriggers;
  vector<Sound*>            * allSounds;
#endif
  rtrt::Array1<Object*>     * allObsOfInterest;
  rtrt::Array1<Object*>     * allAnimateObjects;
  rtrt::Array1<Object*>     * allDynamicBBoxObjects;
  rtrt::Array1<DpyBase*>    * allDisplays;
  rtrt::Array1<DpyBase*>    * allAux_displays;
  rtrt::Array1<Material*>   * allMaterials;
  vector< string >          * allRouteNames_;
  vector< string >          * allRoomsForRoutes_;
  Group *sg = new Group;
  rtrt::Array1<ShadowBase*> * allShadows;

  for (s=0; s<nscenes; s++) {
     std::cerr << "\n\n\n\n==========================================================\n"
               << "Creating scene " << argv[scene_first_arg[s]] 
	       << "\n=========================================================="
	       << "\n";		   
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

    if( s == 0 ) 
      {
#if !defined(linux)
	allTriggers = &(scene[0]->getTriggers());
	allSounds = &(scene[0]->getSounds());
#endif
	allObsOfInterest = &(scene[0]->getObjectsOfInterest());
	allAnimateObjects = &(scene[0]->getAnimateObjects());
	allDynamicBBoxObjects = &(scene[0]->getDynBBoxObjs());
	allDisplays = &(scene[0]->getDisplays());
	allAux_displays = &(scene[0]->getAuxDisplays());
	allMaterials = &(scene[0]->getMaterials());
	allRouteNames_ = &(scene[0]->getRouteNames());
	allRoomsForRoutes_ = &(scene[0]->getRoomsForRoutes());
        allShadows = &(scene[0]->getShadows());
        Object *so = scene[0]->get_shadow_object();
        if (so) sg->objs.add(so);
      }
    else
      {
#if !defined(linux)
	vector<Trigger*> & triggers = scene[s]->getTriggers();
	for( int cnt = 0; cnt < triggers.size(); cnt++ ) {
	  allTriggers->push_back( triggers[cnt] );
	}
	vector<Sound*> & sounds = scene[s]->getSounds();
	for( int cnt = 0; cnt < sounds.size(); cnt++ ) {
	  allSounds->push_back( sounds[cnt] );
	}
#endif
	rtrt::Array1<Object*> & obsOfInterest=scene[s]->getObjectsOfInterest();
	for( int cnt = 0; cnt < obsOfInterest.size(); cnt++ ) {
	  allObsOfInterest->add( obsOfInterest[cnt] );
	}
	rtrt::Array1<Object*> & animateObjs=scene[s]->getAnimateObjects();
	for( int cnt = 0; cnt < animateObjs.size(); cnt++ ) {
	  allAnimateObjects->add( animateObjs[cnt] );
	}
	rtrt::Array1<Object*> & dynBBoxObjs=scene[s]->getDynBBoxObjs();
	for( int cnt = 0; cnt < dynBBoxObjs.size(); cnt++ ) {
	  allDynamicBBoxObjects->add( dynBBoxObjs[cnt] );
	}
	rtrt::Array1<DpyBase*> & displays=scene[s]->getDisplays();
	for( int cnt = 0; cnt < displays.size(); cnt++ ) {
	  allDisplays->add( displays[cnt] );
	}
	rtrt::Array1<DpyBase*> & auxDisplays=scene[s]->getAuxDisplays();
	for( int cnt = 0; cnt < auxDisplays.size(); cnt++ ) {
	  allAux_displays->add( auxDisplays[cnt] );
	}
	rtrt::Array1<Material*> & materials=scene[s]->getMaterials();
	for( int cnt = 0; cnt < materials.size(); cnt++ ) {
	  allMaterials->add( materials[cnt] );
	}
	vector<string> & routeNames = scene[s]->getRouteNames();
	for( int cnt = 0; cnt < routeNames.size(); cnt++ ) {
	  allRouteNames_->push_back( routeNames[cnt] );
	}
	vector<string> & routeRoomNames = scene[s]->getRoomsForRoutes();
	for( int cnt = 0; cnt < routeRoomNames.size(); cnt++ ) {
	  allRouteNames_->push_back( routeRoomNames[cnt] );
	}
	rtrt::Array1<ShadowBase*> & shadows=scene[s]->getShadows();
	for( int cnt = 0; cnt < shadows.size(); cnt++ ) {
	  allShadows->add( shadows[cnt] );
	}
        Object *so = scene[s]->get_shadow_object();
        if (so) sg->objs.add(so);
      }
    
    if (strncmp(argv[scene_first_arg[s]], "scenes/sea", strlen("scenes/sea")) == 0) {
      std::cerr << "\n\n"
		<< " ==== Adding scene's objs to CellGroup WITHOUT a bbox ==="
		<< "\n\n";
      g->add_non_bbox_obj(scene[s]->get_object());
    } else if (strncmp(argv[scene_first_arg[s]], "scenes/basic-sea", strlen("scenes/basic-sea")) == 0) {
      std::cerr << "\n\n"
		<< " ==== Adding scene's objs to CellGroup WITHOUT a bbox ==="
		<< "\n\n";
      g->add_non_bbox_obj(scene[s]->get_object());
    } else {
      std::cerr << "\n\n"
		<< " ==== Adding scene's objs to CellGroup WITH a bbox ==="
		<< "\n\n";
      g->add_bbox_obj(scene[s]->get_object());
    }

    if (s!=0) {
      for (int l=0; l<scene[s]->nlights(); l++)
	scene[0]->add_light(scene[s]->light(l));
      for (int l=0; l<scene[s]->nPerMatlLights(); l++)
	scene[0]->add_per_matl_light(scene[s]->per_matl_light(l));
    }

    std::cerr << "==========================================================\n"
              << "Finished creating scene " << argv[scene_first_arg[s]] 
	      << "\n=========================================================="
	      << "\n\n\n\n";
  }

  cout << "Total number of triggers is: " << scene[0]->getTriggers().size()
       << "\n";

  scene[0]->set_object(g);
  if (sg->objs.size()) scene[0]->shadowobj=sg;
  scene[0]->set_cup(Color(.8,.8,.8));
  scene[0]->set_cdown(Color(.4,.4,.4));
  return scene[0];
}







