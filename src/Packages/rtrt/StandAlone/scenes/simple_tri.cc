

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Tri.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;
using namespace std;
using namespace SCIRun;

static const double SCALE = 1./3.;
static const double BV_RADIUS = 1.0;

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
	    cerr << " -size n   - Sets depth of sphereflake\n";
	    cerr << " -light r  - Sets radius of light source for soft shadows\n";
	    return 0;
	}
    }

    Camera cam(Point(32.4973, 16.8985, 8.92619), Point(3.5, 10, 0),
                        Vector(-0.0160856, -0.345967, 0.938109), 40);

    Point p0(-1,0,-.5);
    Point p1(0,-.5,0);
    Point p2(0,.5,0);

    Point p01(1,0,-.5);
    Point p11(0,-.5,0);
    Point p21(0,.5,0);

    Vector fn0 = Cross(p1-p0,p2-p0);
    Vector fn2 = -Cross(p11-p01,p21-p01);
    Vector fn1 = .5*(fn0+fn2);

    Material* mat = new Phong(Color(0,.4,0),Color(.2,.2,.2), 30);

    Object* obj=new Tri(mat,p0,p1,p2,fn0,fn1,fn1);
    Object* obj1=new Tri(mat,p01,p11,p21,fn2,fn1,fn1);

    Group* g = new Group();
    g->add(obj);
    g->add(obj1);

    double ambient_scale=1.0;
    Color bgcolor(0.1, 0.2, 0.45);
    Color cdown(0.82, 0.62, 0.62);
    Color cup(0.1, 0.3, 0.8);


    rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 3) );
    Scene* scene=new Scene(g, cam,
			   bgcolor, cdown, cup, groundplane,
			   ambient_scale);
    scene->add_light(new Light(Point(1,0,3), Color(.5,.5,.5), light_radius));

    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(0,0,1)) );


    scene->select_shadow_mode( Hard_Shadows );
    return scene;
}
