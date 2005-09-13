

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/CutPlane.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;

using SCIRun::Thread;


extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
    Camera cam(Point(0,3,20), Point(0.0,0,0),
                        Vector(0,1,0), 40);

    double ambient_scale=1.0;
    Color bgcolor(0.1, 0.2, 0.45);
    Color cdown(0.82, 0.62, 0.62);
    Color cup(0.1, 0.3, 0.8);


    rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 3) );

    Group *g = new Group();
    
    Material *mtl = new LambertianMaterial(Color(0.5,0.5,0.5));
    Sphere *s = new Sphere(mtl, Point(0,0,0), 1.25);
    Box *b = new Box(mtl, Point(-1,-1,-1), Point(1,1,1));
    InstanceWrapperObject *sw = new InstanceWrapperObject(s);
    InstanceWrapperObject *bw = new InstanceWrapperObject(b);

    Transform *t = new Transform();
    t->pre_translate(Vector(-2,0,3));
    Instance *si = new Instance(sw, t);
    t->pre_translate(Vector(4,0,0));
    Instance *bi = new Instance(bw, t);


    Transform *st = new Transform();
    st->pre_scale(Vector(0.5,0.5,0.5));
    st->pre_translate(Vector(-2,0,6));
    SpinningInstance *ssi  = new SpinningInstance(sw, st, Point(-2,0,6.5), Vector(1,1,0), 0.5);
    st->pre_translate(Vector(4,0,0));
    SpinningInstance *bsi = new SpinningInstance(bw, st, Point(2,0,6.5), Vector(1,1,0), 0.5);

    
    g->add(s);
    g->add(b);
    g->add(si);
    g->add(bi);
    g->add(ssi);
    g->add(bsi);

    Scene* scene=new Scene(g, cam,
			   bgcolor, cdown, cup, groundplane,
			   ambient_scale, Arc_Ambient);

    scene->add_light(new Light(Point(0,10,0), Color(1,1,1), 0.8));

    scene->set_background_ptr( new LinearBackground(
                               Color(0.2, 0.4, 0.9),
                               Color(0.0,0.0,0.0),
                               Vector(0,0,1)) );


  
    scene->addObjectOfInterest( ssi, true);
    scene->addObjectOfInterest( bsi, true);
    scene->select_shadow_mode( Single_Soft_Shadow );
    return scene;
}
