#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/UVCylinder.h>
#include <Packages/rtrt/Core/UVCylinderArc.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace rtrt;

#define CYLTEXSCALEX  .06
#define CYLTEXSCALEY  .3
#define DOORWIDTH     .05
#define DOORHEIGHT    2.5
#define ROOMHEIGHT    10
#define HEIGHTRATIO   (DOORHEIGHT/ROOMHEIGHT)
#define ROOMCENTER    9,9
#define ROOMRADIUS    4
#define WALLTHICKNESS .1

#define INSCILAB 1

extern "C"
Scene* make_scene(int /*argc*/, char* /*argv*/[], int /*nworkers*/)
{
  Camera cam( Point(0,0,0), Point( 0,-1,0 ), Vector(0,0,1), 45.0 );

   ImageMaterial* bookcoverimg = 
       new ImageMaterial(1,
#if INSCILAB
                         "/usr/sci/data/Geometry/textures/i3d97.smaller.gamma",
#else
                         "/home/moulding/i3d97.smaller.gamma",
#endif
                         ImageMaterial::Tile,
                         ImageMaterial::Tile, 1,
                         Color(1,1,1), 4000);

  ImageMaterial *matl0 = new ImageMaterial(
#if INSCILAB                                         
                                           "/home/cs/moulding/holo.ppm",
#else
                                           "/home/moulding/holo.ppm",
#endif
                                           ImageMaterial::Tile,
                                           ImageMaterial::Tile,
                                           4,Color(.5,.5,.5),40,0,0);

  matl0->flip();
  bookcoverimg->flip();
  Material *white = new LambertianMaterial(Color(1,1,1));
  
  UVCylinderArc* uvcylarc0 = new UVCylinderArc(matl0,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS);
  UVCylinderArc* uvcylarc1 = new UVCylinderArc(matl0,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS);
  UVCylinderArc* uvcylarc2 = new UVCylinderArc(white,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS+WALLTHICKNESS);
  UVCylinderArc* uvcylarc3 = new UVCylinderArc(white,Point(ROOMCENTER,0),
                                               Point(ROOMCENTER,DOORHEIGHT),
                                               ROOMRADIUS+WALLTHICKNESS);
  UVCylinder* uvcyl0 = new UVCylinder(matl0,Point(ROOMCENTER,DOORHEIGHT),
                                      Point(ROOMCENTER,ROOMHEIGHT),ROOMRADIUS);
  UVCylinder* uvcyl1 = new UVCylinder(white,Point(ROOMCENTER,DOORHEIGHT),
                                      Point(ROOMCENTER,ROOMHEIGHT),
                                      ROOMRADIUS+WALLTHICKNESS);

  Rect* floor = new Rect(matl0,Point(ROOMCENTER,0),Vector(ROOMRADIUS,0,0),
                          Vector(0,ROOMRADIUS,0));
  Object* obj1 = new Sphere(matl0,Point(5,5,0),1);
  Point p1(ROOMCENTER,.2);
  Parallelogram *bookcover = new Parallelogram(bookcoverimg,p1,
                                               Vector(0,-1,0),
                                               Vector(-.774,0,0));
  
  uvcylarc0->set_arc((DOORWIDTH)*M_PI,(.5-DOORWIDTH)*M_PI);
  uvcylarc1->set_arc((.5+DOORWIDTH)*M_PI,(2-DOORWIDTH)*M_PI);
  uvcylarc2->set_arc((DOORWIDTH)*M_PI,(.5-DOORWIDTH)*M_PI);
  uvcylarc3->set_arc((.5+DOORWIDTH)*M_PI,(2-DOORWIDTH)*M_PI);

  uvcylarc0->set_tex_scale(Vector(CYLTEXSCALEX,
                                  CYLTEXSCALEY/HEIGHTRATIO,0));
  uvcylarc1->set_tex_scale(Vector(CYLTEXSCALEX,
                                  CYLTEXSCALEY/HEIGHTRATIO,0));
  uvcyl0->set_tex_scale(Vector(CYLTEXSCALEX,CYLTEXSCALEY,0));
  floor->set_tex_scale(Vector(5,5,0));
  
  Group * group = new Group();
  group->add( uvcylarc0 );
  group->add( uvcylarc1 );
  group->add( uvcylarc2 );
  group->add( uvcylarc3 );
  group->add( uvcyl0 );
  group->add( uvcyl1 );
  group->add( floor );
  group->add( obj1 );
  group->add( bookcover );

  double ambient_scale=1.0;
  Color bgcolor(0.1, 0.2, 0.45);
  Color cdown(0.82, 0.62, 0.62);
  Color cup(0.1, 0.3, 0.8);

  rtrt::Plane groundplane ( Point(0, 0, 0), Vector(0, 0, 1) );
  Scene* scene=new Scene(group, cam,
                         bgcolor, cdown, cup, groundplane,
                         ambient_scale);
  scene->add_light( new Light(Point(ROOMCENTER,9.5), Color(1,1,1), 0.8) );
  scene->ambient_hack = true;

  scene->set_background_ptr( new LinearBackground( Color(1.0, 1.0, 1.0),
                                                   Color(0.0,0.0,0.0),
                                                   Vector(0,0,1)) );
/*
  scene->select_shadow_mode("hard");
*/
  return scene;
}


