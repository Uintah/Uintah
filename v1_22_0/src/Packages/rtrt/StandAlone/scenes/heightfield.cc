
#include <Packages/rtrt/Core/BrickArray2.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/rtrt.h>

#include <iostream>
#include <fstream>

#include <math.h>
#include <string.h>

using namespace rtrt;

extern "C" 
Scene* make_scene(int argc, char* argv[])
{
    char* file=0;
    int depth=3;
    bool shownodes=false;
    for(int i=1;i<argc;i++){
       if(strcmp(argv[i], "-depth")==0){
	    i++;
	    depth=atoi(argv[i]);
       } else if(strcmp(argv[i], "-shownodes")==0){
	  shownodes=true;
	} else {
	    if(file){
		cerr << "Unknown option: " << argv[i] << '\n';
		cerr << "Valid options for scene: " << argv[0] << '\n';
		cerr << " -rate\n";
		cerr << " -depth\n";
		return 0;
	    }
	    file=argv[i];
	}
    }

    Camera cam(Point(5,0,0), Point(0,0,0),
	       Vector(0,0,1), 60);

    Color surf(1.00000, 0.0, 0.00);
    //Material* matl0=new PhongMaterial(surf, 1, .3, 40);
    //Material* matl0 = new LambertianMaterial(surf);
    Material *matl0 = new Phong(Color(.63,.51,.5),Color(.3,.3,.3),400);
    //Material* matl0=new Phong(Color(.6,.6,0),Color(.5,.5,.5),30);
#if 0
    Material* matl0=new CrowMarble(2, 
                          Vector(2,1,0),
                          Color(0.5,0.6,0.6),
                          Color(0.4,0.55,0.52),
                          Color(0.035,0.045,0.042)
                                      );
#endif
    cerr << "Reading " << file << "\n";
    Heightfield<BrickArray2<float>, Array2<HMCell<float> > >* hf
       =new Heightfield<BrickArray2<float>, Array2<HMCell<float> > >
           (matl0, file, depth, 16);

    Object* obj = hf;
    if(shownodes){
       Material* matl1=new Phong (Color(.6,.6,0),Color(.5,.5,.5),30);
       BrickArray2<float>& data = hf->blockdata;
       Group* g = new Group();
       double rad  =Min(hf->sdiag.x(), hf->sdiag.y(), hf->sdiag.z())*0.05;
       for(int i=0;i<data.dim1();i++){
	  for(int j=0;j<data.dim2();j++){
	     double h = data(i,j);
	     Point p = hf->min+Vector(i,j,h)*hf->sdiag;
	     p.z(h);
	     g->add(new Sphere(matl1, p, rad));
	  }
       }
       g->add(hf);
       obj=g;
    }

    double bgscale=0.3;
    Color groundcolor(0,0,0);
    Color averagelight(1,1,1);
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(obj, cam,
			   bgcolor, groundcolor*averagelight, bgcolor, groundplane, 
			   ambient_scale);
    scene->add_light(new Light(Point(5,-3,3), Color(1,1,.8)*2, 0));
    scene->select_shadow_mode( No_Shadows );
    scene->maxdepth=0;

    return scene;
}
