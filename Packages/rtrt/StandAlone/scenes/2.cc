
#include <Packages/rtrt/Core/BouncingSphere.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;
using namespace std;

extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
    double light_radius=0.8;
    for(int i=1;i<argc;i++){
	if(strcmp(argv[i], "-light")==0){
	    i++;
	    light_radius=atof(argv[i]);
	} else {
	    cerr << "Unknown option: " << argv[i] << '\n';
	    cerr << "Valid options for scene: " << argv[0] << '\n';
	    cerr << " -light r    - set the radius of the light source to r\n";
	    return 0;
	}
    }

    Camera cam(Point(4,4,4), Point(0,0,0),
	       Vector(0,0,1), 60.0);

    //Color groundcolor(.75, .55, .4);
    //Color averagelight(0.9,0.6,0.2);
    double ambient_scale=1;

    Color bgcolor(0.05, 0.1, 0.3);

    //Material* matl0=new LambertianMaterial(Color(0.8,0.8,0.8));
    Material* matl2=new DielectricMaterial(1.5, 1.0, 0.04, 100.0, Color(.85, .97, .9), Color(1,1,1));
    Material* matl3=new DielectricMaterial(1.33333, 1.0);
    Material* matl20=new DielectricMaterial(1.0, 1.5, 0.04, 100.0,  Color(1,1,1),  Color(.85, .97, .9) );
    Material* matl30=new DielectricMaterial(1.0, 1.3333);
    Material* matl4 = new MetalMaterial(Color(0.7,0.73,0.8));

    Material* matl1=new Checker(new LambertianMaterial(Color(1.0,1.0,1.0)),
				new LambertianMaterial(Color(.7,.3,.1)),
				Vector(1,1,0), Vector(-1,1,0));
    Object* obj1=new Rect(matl1, Point(0,0,0), Vector(6,0,0), Vector(0,6,0));

    Group* group=new Group();
    group->add(obj1);

    BouncingSphere * bs1, * bs2, *bs3, *bs4, *bs5, *bs6;

    bs1 = new BouncingSphere(matl4, Point(0,0,.4), .5, Vector(0,0,1));
    group->add(bs1);

    bs2 = new BouncingSphere(matl2, Point(0,0,1.4), .5, Vector(0,0,1.2));
    group->add(bs2);

    bs3 = new BouncingSphere(matl3, Point(0,0,2.4), .5, Vector(0,0,1.4));
    group->add(bs3);

    bs4 = new BouncingSphere(matl2, Point(0,0,3.4), .5, Vector(0,0,1.6));
    group->add(bs4);

    double thickness=0.05;
    bs5 = new BouncingSphere(matl20, Point(0,0,1.4), .5-thickness, Vector(0,0,1.2));
    group->add(bs5);

    double thickness2=0.0005;
    bs6 = new BouncingSphere(matl30, Point(0,0,2.4), .5-thickness2, Vector(0,0,1.4));
    group->add(bs6);


    Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 4) );
    Color cdown(0.82, 0.62, 0.62);
    Color cup(0.1, 0.3, 0.8);
    Scene* scene=new Scene(group, cam,
			   bgcolor, cdown, cup, groundplane, 
			   ambient_scale, Arc_Ambient);
    scene->add_light(new Light(Point(50,30,60), Color(0.9,0.6,0.2), light_radius));
    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(0,0,1)) );

    scene->select_shadow_mode(Hard_Shadows);

    scene->addObjectOfInterest( bs1, true );
    scene->addObjectOfInterest( bs2, true );
    scene->addObjectOfInterest( bs3, true );
    scene->addObjectOfInterest( bs4, true );
    scene->addObjectOfInterest( bs5, true );
    scene->addObjectOfInterest( bs6, true );
    scene->maxdepth=20;

    return scene;
}
