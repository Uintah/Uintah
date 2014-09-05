#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <Packages/rtrt/Core/Point4D.h>
#include <Packages/rtrt/Core/CrowMarble.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Mesh.h>
#include <Packages/rtrt/Core/ASEReader.h>
#include <Packages/rtrt/Core/ObjReader.h>
#include <Packages/rtrt/Core/Bezier.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Speckle.h>
#include <Packages/rtrt/Core/Box.h>
#include <Packages/rtrt/Core/CoupledMaterial.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/MetalMaterial.h>
#include <Packages/rtrt/Core/Rect.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Core/Math/MinMax.h>
#include <Packages/rtrt/Core/Tri.h>
#include <Packages/rtrt/Core/TexturedTri.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/UVCylinderArc.h>
#include <Packages/rtrt/Core/UVCylinder.h>


#include <Core/Thread/Thread.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/HVolumeBrick16.h>
#include <Packages/rtrt/Core/MIPHVB16.h>
#include <Packages/rtrt/Core/CutVolumeDpy.h>
#include <Packages/rtrt/Core/CutPlaneDpy.h>
#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/CutMaterial.h>
#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Instance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/SpinningInstance.h>
#include <Packages/rtrt/Core/DynamicInstance.h>
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <Packages/rtrt/Core/SpinningInstance.h>

using namespace rtrt;
using namespace std;
using SCIRun::Thread;

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }

// Start inside:
//  Point Eye(-5.85, 6.2, 2.0);
//  Point Lookat(-13.5, 13.5, 2.0);
//  Vector Up(0,0,1);
//  double fov=60;

// Start outside:
  Point Eye(-10.9055, -0.629515, 1.56536);
  Point Lookat(-8.07587, 15.7687, 1.56536);
  Vector Up(0, 0, 1);
  double fov=60;

// Just table:
//  Point Eye(-7.64928, 6.97951, 1.00543);
//  Point Lookat(-19.9299, 16.5929, -2.58537);
//  Vector Up(0, 0, 1);
//  double fov=35;

  Camera cam(Eye,Lookat,Up,fov);

  Point center(-8, 8, 0);
  Vector north(0,1,0);
  Vector east(1,0,0);
  Vector up(0,0,1);
  double east_west_wall_length=8;
  double north_south_wall_length=8;
  double wall_height=4;
  double door_height=2.3;
  double door_width=1.7;
  double door_inset_distance=1.15;
  double wall_thickness=0.2;
  double fc_thickness=0.001;

  Group* north_wall=new Group();
  Group* west_wall=new Group();
  Group* south_wall=new Group();
  Group* east_wall=new Group();
  Group* table=new Group();
  Group* ceiling_floor=new Group();

  ImageMaterial *stucco = new ImageMaterial("/opt/SCIRun/data/Geometry/textures/science-room/stucco.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0, 0, true);

  Point north_wall_center(center+east_west_wall_length/2*north+
			  wall_height/2*up);
  north_wall->add(new Rect(stucco, north_wall_center,
			   4*east, 2*up));
  north_wall->add(new Rect(stucco, north_wall_center+north*wall_thickness,
			   (4+wall_thickness)*east,(2+fc_thickness)*up));
  Point west_wall_center(center-north_south_wall_length/2*east+
			 wall_height/2*up);
  west_wall->add(new Rect(stucco, west_wall_center,
			  4*north, 2*up));
  west_wall->add(new Rect(stucco, west_wall_center-east*wall_thickness,
			  (4+wall_thickness)*north,(2+fc_thickness)*up));

//  south_wall->add(new Rect(white, Point(-8, 4, 2), 
//		       Vector(4, 0, 0), Vector(0, 0, 2)));

  // doorway cut out of South wall for W. tube: attaches to Graphic Museum scene

  Point south_wall_center(center-east_west_wall_length/2*north+
			  wall_height/2*up);
  Point south_floor_west_corner(center-east_west_wall_length/2*north-
				north_south_wall_length/2*east);
  Point south_floor_east_corner(south_floor_west_corner+
				north_south_wall_length*east);
  Point south_ceiling_west_corner(south_floor_west_corner+
				  wall_height*up);
  Point south_ceiling_east_corner(south_floor_east_corner+
				  wall_height*up);

  Point south_floor_west_out_corner(south_floor_west_corner-
				    north*wall_thickness-
				    east*wall_thickness-
				    up*fc_thickness);
  Point south_floor_east_out_corner(south_floor_east_corner-
				    north*wall_thickness+
				    east*wall_thickness-
				    up*fc_thickness);
  Point south_ceiling_west_out_corner(south_ceiling_west_corner-
				    north*wall_thickness-
				    east*wall_thickness+
				    up*fc_thickness);
  Point south_ceiling_east_out_corner(south_ceiling_east_corner-
				    north*wall_thickness+
				    east*wall_thickness+
				    up*fc_thickness);
				      
  Point north_floor_west_corner(center+east_west_wall_length/2*north-
				north_south_wall_length/2*east);
  Point north_floor_east_corner(north_floor_west_corner+
				north_south_wall_length*east);
  Point north_ceiling_west_corner(north_floor_west_corner+
				  wall_height*up);
  Point north_ceiling_east_corner(north_floor_east_corner+
				  wall_height*up);

  Point north_floor_west_out_corner(north_floor_west_corner+
				    north*wall_thickness-
				    east*wall_thickness-
				    up*fc_thickness);
  Point north_floor_east_out_corner(north_floor_east_corner+
				    north*wall_thickness+
				    east*wall_thickness-
				    up*fc_thickness);
  Point north_ceiling_west_out_corner(north_ceiling_west_corner+
				    north*wall_thickness-
				    east*wall_thickness+
				    up*fc_thickness);
  Point north_ceiling_east_out_corner(north_ceiling_east_corner+
				    north*wall_thickness+
				    east*wall_thickness+
				    up*fc_thickness);

// rectangle that goes across the whole wall, over the doorway
//
//    1 -------------------- 6 (from outside)
//      |                  |
//      |  7  8            |
//      |  ----            |
//      |  |  |            |
//      |  |  |            |
//      |-------------------
//     2   3  4             5

  Point p1,p2,p3,p4,p5,p6,p7,p8;
  Point p1t,p2t,p3t,p4t,p5t,p6t,p7t,p8t;
  Point p1o,p2o,p3o,p4o,p5o,p6o,p7o,p8o;
  TexturedTri *tri;

  p1=Point(south_ceiling_west_corner);
  p1t=Point(1,1,0);
  p2=Point(south_floor_west_corner);
  p2t=Point(1,0,0);
  p3=Point(south_floor_west_corner+door_inset_distance*east);
  p3t=Point(1-door_inset_distance/north_south_wall_length,0,0);
  p4=Point(p3+door_width*east);
  p4t=Point(1-(door_inset_distance+door_width)/north_south_wall_length,0,0);
  p5=Point(south_floor_east_corner);
  p5t=Point(0,0,0);
  p6=Point(south_ceiling_east_corner);
  p6t=Point(0,1,0);
  p7=Point(p3+door_height*up);
  p7t=Point(p3t.x(),door_height/wall_height,0);
  p8=Point(p4+door_height*up);
  p8t=Point(p4t.x(),door_height/wall_height,0);

  p1o=Point(p1-north*wall_thickness-east*wall_thickness+up*fc_thickness);
  p2o=Point(p2-north*wall_thickness-east*wall_thickness-up*fc_thickness);
  p3o=Point(p3-north*wall_thickness-up*fc_thickness);
  p4o=Point(p4-north*wall_thickness-up*fc_thickness);
  p5o=Point(p5-north*wall_thickness+east*wall_thickness-up*fc_thickness);
  p6o=Point(p6-north*wall_thickness+east*wall_thickness+up*fc_thickness);
  p7o=Point(p7-north*wall_thickness);
  p8o=Point(p8-north*wall_thickness);

  Point s3(p3),s4(p4),s7(p7),s8(p8);
  Point s3o(p3o), s4o(p4o), s7o(p7o), s8o(p8o);

  tri = new TexturedTri(stucco, p1, p2, p3);
  tri->set_texcoords(p1t,p2t,p3t);
  south_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p2o, p3o);
  tri->set_texcoords(p1t,p2t,p3t);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p3, p7);
  tri->set_texcoords(p1t,p3t,p7t);
  south_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p3o, p7o);
  tri->set_texcoords(p1t,p3t,p7t);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p7, p8);
  tri->set_texcoords(p1t,p7t,p8t);
  south_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p7o, p8o);
  tri->set_texcoords(p1t,p7t,p8t);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p8, p6);
  tri->set_texcoords(p1t,p8t,p6t);
  south_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p8o, p6o);
  tri->set_texcoords(p1t,p8t,p6t);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p5, p8, p6);
  tri->set_texcoords(p5t,p8t,p6t);
  south_wall->add(tri);
  tri = new TexturedTri(stucco, p5o, p8o, p6o);
  tri->set_texcoords(p5t,p8t,p6t);
  south_wall->add(tri);

  tri = new TexturedTri(stucco, p4, p8, p5);
  tri->set_texcoords(p4t,p8t,p5t);
  south_wall->add(tri);
  tri = new TexturedTri(stucco, p4o, p8o, p5o);
  tri->set_texcoords(p4t,p8t,p5t);
  south_wall->add(tri);

  p1=Point(north_ceiling_east_corner);
  p1t=Point(0,1,0);
  p2=Point(north_floor_east_corner);
  p2t=Point(0,0,0);
  p3=Point(north_floor_east_corner-door_inset_distance*north);
  p3t=Point(door_inset_distance/east_west_wall_length,0,0);
  p4=Point(p3-door_width*north);
  p4t=Point((door_inset_distance+door_width)/east_west_wall_length,0,0);
  p5=Point(south_floor_east_corner);
  p5t=Point(1,0,0);
  p6=Point(south_ceiling_east_corner);
  p6t=Point(1,1,0);
  p7=Point(p3+door_height*up);
  p7t=Point(p3t.x(),door_height/wall_height,0);
  p8=Point(p4+door_height*up);
  p8t=Point(p4t.x(),door_height/wall_height,0);

  p1o=Point(p1+north*wall_thickness+east*wall_thickness+up*fc_thickness);
  p2o=Point(p2+north*wall_thickness+east*wall_thickness-up*fc_thickness);
  p3o=Point(p3+east*wall_thickness-up*fc_thickness);
  p4o=Point(p4+east*wall_thickness-up*fc_thickness);
  p5o=Point(p5-north*wall_thickness+east*wall_thickness-up*fc_thickness);
  p6o=Point(p6-north*wall_thickness+east*wall_thickness+up*fc_thickness);
  p7o=Point(p7+east*wall_thickness);
  p8o=Point(p8+east*wall_thickness);

  Point e3(p3), e4(p4), e7(p7), e8(p8); 
  Point e3o(p3o), e4o(p4o), e7o(p7o), e8o(p8o);
  
  tri = new TexturedTri(stucco, p1, p2, p3);
  tri->set_texcoords(p1t,p2t,p3t);
  east_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p2o, p3o);
  tri->set_texcoords(p1t,p2t,p3t);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p3, p7);
  tri->set_texcoords(p1t,p3t,p7t);
  east_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p3o, p7o);
  tri->set_texcoords(p1t,p3t,p7t);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p7, p8);
  tri->set_texcoords(p1t,p7t,p8t);
  east_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p7o, p8o);
  tri->set_texcoords(p1t,p7t,p8t);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p1, p8, p6);
  tri->set_texcoords(p1t,p8t,p6t);
  east_wall->add(tri);
  tri = new TexturedTri(stucco, p1o, p8o, p6o);
  tri->set_texcoords(p1t,p8t,p6t);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p5, p8, p6);
  tri->set_texcoords(p5t,p8t,p6t);
  east_wall->add(tri);
  tri = new TexturedTri(stucco, p5o, p8o, p6o);
  tri->set_texcoords(p5t,p8t,p6t);
  east_wall->add(tri);

  tri = new TexturedTri(stucco, p4, p8, p5);
  tri->set_texcoords(p4t,p8t,p5t);
  east_wall->add(tri);
  tri = new TexturedTri(stucco, p4o, p8o, p5o);
  tri->set_texcoords(p4t,p8t,p5t);
  east_wall->add(tri);

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));
  Material *bb_matl = new Phong(Color(0.45,0.45,0.45), Color(0.3,0.3,0.3), 20, 0);
  UVCylinderArc *uvc;

  uvc=new UVCylinderArc(bb_matl, north_floor_west_corner+north*.05,
			north_floor_east_corner+north*.05, 0.1);
  uvc->set_arc(0,M_PI/2);
  north_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_floor_west_corner-east*.05,
			south_floor_west_corner-east*.05, 0.1);
  uvc->set_arc(M_PI,3*M_PI/2);
  west_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_ceiling_west_corner+north*.05,
			north_ceiling_east_corner+north*.05, 0.1);
  uvc->set_arc(M_PI/2,M_PI);
  north_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_ceiling_west_corner-east*.05,
			south_ceiling_west_corner-east*.05, 0.1);
  uvc->set_arc(M_PI/2, M_PI);
  west_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, south_ceiling_west_corner-north*.05,
			south_ceiling_east_corner-north*.05, 0.1);
  uvc->set_arc(M_PI,3*M_PI/2);
  south_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, north_ceiling_east_corner+east*.05,
			south_ceiling_east_corner+east*.05, 0.1);
  uvc->set_arc(0,M_PI/2);
  east_wall->add(uvc);

  uvc=new UVCylinderArc(bb_matl, south_floor_west_corner-north*.05,
			s3-north*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);
  south_wall->add(uvc);
  uvc=new UVCylinderArc(bb_matl, s4-north*.05, 
			south_floor_east_corner-north*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);
  south_wall->add(uvc);

  Material *black = new Phong(Color(0.1,0.1,0.1), Color(0.3,0.3,0.3), 20, 0);
  south_wall->add(new Box(black,s3-north*wall_thickness-north*.1-east*.1,
			  s7+north*.1+east*.1+up*.1));
  south_wall->add(new Box(black,s7-north*wall_thickness-north*.1-east*.1-up*.1,
			  s8+north*.1+east*.1+up*.1));
  south_wall->add(new Box(black,s4-north*wall_thickness-north*.1-east*.1,
			  s8+north*.1+east*.1+up*.1));
  
  uvc=new UVCylinderArc(bb_matl, north_floor_east_corner+east*.05,
			e3+east*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);
  east_wall->add(uvc);
  uvc=new UVCylinderArc(bb_matl, e4+east*.05, 
			south_floor_east_corner+east*.05, 0.1);
  uvc->set_arc(3*M_PI/2, 2*M_PI);  
  east_wall->add(uvc);

  east_wall->add(new Box(black,e3-north*.1-east*.1, 
			 e7+north*.1+east*.1+up*.1+east*wall_thickness));
  east_wall->add(new Box(black,e8-north*.1-east*.1-up*.1,
			 e7+north*.1+east*.1+up*.1+east*wall_thickness));
  east_wall->add(new Box(black,e4-north*.1-east*.1,
			 e8+north*.1+east*.1+up*.1+east*wall_thickness));

  east_wall->add(new UVCylinderArc(bb_matl, e3+east*.05, e7+east*.05, 0.1));
  east_wall->add(new UVCylinderArc(bb_matl, e7+east*.05, e8+east*.05, 0.1));
  east_wall->add(new UVCylinderArc(bb_matl, e8+east*.05, e4+east*.05, 0.1));
  
  // add the ceiling
  ceiling_floor->add(new Rect(white, Point(-8, 8, 4),
		       Vector(4, 0, 0), Vector(0, 4, 0)));
  ceiling_floor->add(new Rect(white, Point(-8, 8, 4)+up*fc_thickness,
			      Vector(4+wall_thickness, 0, 0), 
			      Vector(0, 4+wall_thickness, 0)));

  // table top
  ImageMaterial *cement_floor = 
    new ImageMaterial("/opt/SCIRun/data/Geometry/textures/science-room/cement-floor.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp, 0,
		      Color(0,0,0), 0);
  Object* floor=new Rect(cement_floor, Point(-8, 8, 0),
			 Vector(4, 0, 0), Vector(0, 4, 0));
  ceiling_floor->add(floor);

  ImageMaterial *cement_pedestal = 
    new ImageMaterial("/opt/SCIRun/data/Geometry/textures/science-room/cement-pedestal.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp, 0,
		      Color(0,0,0), 0);
  table->add(new UVCylinder(cement_pedestal, center,
			    center+Vector(0,0,0.5), 1.5));
  Material *silver = new MetalMaterial(Color(0.5,0.5,0.5), 12);
  table->add(new Disc(silver, center+Vector(0,0,0.5),
		      Vector(0,0,1), 1.5));

  Group *g = new Group();

  g->add(ceiling_floor);
  g->add(north_wall);
  g->add(west_wall);
  g->add(south_wall);
  g->add(east_wall);
  g->add(table);

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);

  Scene *scene = new Scene(new Grid(g, 16),
			   cam, bgcolor, cdown, cup, groundplane, 0.3);
//  Scene *scene = new Scene(g,
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;
  Light *science_room_light0 = new Light(Point(-8, 8, 3.9), Color(.5,.5,.5), 0);
  science_room_light0->name_ = "science room overhead";
  scene->add_light(science_room_light0);
  Light *science_room_light1 = new Light(Point(-11, 11, 3), Color(.5,.5,.5), 0);
  science_room_light1->name_ = "science room corner";
  scene->add_light(science_room_light1);
  scene->animate=true;

  return scene;
}
