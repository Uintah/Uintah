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
#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/HVolume.h>
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

using namespace rtrt;
using namespace std;
using SCIRun::Thread;

#define ADD_VIS_FEM 1
#define ADD_DAVE_HEAD 1
#define ADD_CSAFE_FIRE 1
#define ADD_GEO_DATA 1

void make_walls_and_posters(Group *g, const Point &center) {
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
  Group* ceiling_floor=new Group();

  ImageMaterial *stucco = new ImageMaterial("/usr/sci/data/Geometry/textures/science-room/stucco.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);

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

  Material *gray = new LambertianMaterial(Color(0.3,0.3,0.3));

  double cable_radius=0.02;
  Point north_cable_1_base(north_floor_west_corner+
			   east*east_west_wall_length/4-north*.08);
  Point north_cable_2_base(north_cable_1_base+east_west_wall_length/4*east);
  Point north_cable_3_base(north_cable_2_base+east_west_wall_length/4*east);
  north_wall->add(new Cylinder(gray, north_cable_1_base,
			       north_cable_1_base+up*wall_height, 
			       cable_radius));
  north_wall->add(new Cylinder(gray, north_cable_2_base,
			       north_cable_2_base+up*wall_height, 
			       cable_radius));
  north_wall->add(new Cylinder(gray, north_cable_3_base,
			       north_cable_3_base+up*wall_height, 
			       cable_radius));

  Material* dnaM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/DNA2.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  north_wall->add(new Rect(dnaM, north_cable_1_base+up*2.4-
			   north*(cable_radius+.005), east*.93, up*-1.2));

  Material* hypatiaM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/Hypatia.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  north_wall->add(new Rect(hypatiaM, north_cable_2_base+up*3.0-
			   north*(cable_radius+.005), east*.55, up*-.6));

  Material* emilieM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/emilie_du_chatelet.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  north_wall->add(new Rect(emilieM, north_cable_2_base+up*1.6-
			   north*(cable_radius+.005), east*.45, up*-.55));

  Material* bluesunM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/bluesun.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  north_wall->add(new Rect(bluesunM, north_cable_3_base+up*2.5-
			   north*(cable_radius+.005), east*1.1, up*-1.1));

  Point west_cable_1_base(south_floor_west_corner+
			   north*north_south_wall_length/5+east*.08);
  Point west_cable_2_base(west_cable_1_base+north_south_wall_length/5*north);
  Point west_cable_3_base(west_cable_2_base+north_south_wall_length/5*north);
  Point west_cable_4_base(west_cable_3_base+north_south_wall_length/5*north);
  west_wall->add(new Cylinder(gray, west_cable_1_base,
			      west_cable_1_base+up*wall_height, 
			      cable_radius));
  west_wall->add(new Cylinder(gray, west_cable_2_base,
			      west_cable_2_base+up*wall_height, 
			      cable_radius));
  west_wall->add(new Cylinder(gray, west_cable_3_base,
			      west_cable_3_base+up*wall_height, 
			      cable_radius));
  west_wall->add(new Cylinder(gray, west_cable_4_base,
			      west_cable_4_base+up*wall_height, 
			      cable_radius));

  Material* galileoM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/galileo.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(galileoM, west_cable_1_base+up*2.3+
			  east*(cable_radius+.005), north*.95, up*-1.4));

  Material* brunoM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/bruno.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(brunoM, west_cable_2_base+up*2.95+
			  east*(cable_radius+.005), north*.43, up*-0.55));

  Material* maxwellM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/james_clerk_maxwell.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(maxwellM, west_cable_2_base+up*1.6+
			  east*(cable_radius+.005), north*.4, up*-0.5));

  Material* joeM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/joe_head.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(joeM, west_cable_3_base+up*1.65+
			  east*(cable_radius+.005), north*.7, up*-0.75));

  Material* australM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/australopithecus_boisei.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(australM, west_cable_4_base+up*1.65+
			  east*(cable_radius+.005), north*.62, up*-0.75));

  Material* apolloM =
    new ImageMaterial("/usr/sci/data/Geometry/models/science-room/posters2/Apollo16_lander.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  west_wall->add(new Rect(apolloM, west_cable_3_base+up*3.2+north*.75+
			  east*(cable_radius+.005), north*1.5, up*-0.5));

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
  p1t=Point(0,0,0);
  p2=Point(south_floor_west_corner);
  p2t=Point(0,1,0);
  p3=Point(south_floor_west_corner+door_inset_distance*east);
  p3t=Point(door_inset_distance/north_south_wall_length,1,0);
  p4=Point(p3+door_width*east);
  p4t=Point((door_inset_distance+door_width)/north_south_wall_length,1,0);
  p5=Point(south_floor_east_corner);
  p5t=Point(1,1,0);
  p6=Point(south_ceiling_east_corner);
  p6t=Point(1,0,0);
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
  p1t=Point(0,0,0);
  p2=Point(north_floor_east_corner);
  p2t=Point(0,1,0);
  p3=Point(north_floor_east_corner-door_inset_distance*north);
  p3t=Point(door_inset_distance/east_west_wall_length,1,0);
  p4=Point(p3-door_width*north);
  p4t=Point((door_inset_distance+door_width)/east_west_wall_length,1,0);
  p5=Point(south_floor_east_corner);
  p5t=Point(1,1,0);
  p6=Point(south_ceiling_east_corner);
  p6t=Point(1,0,0);
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
    new ImageMaterial("/usr/sci/data/Geometry/textures/science-room/cement-floor.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp, 0,
		      Color(0,0,0), 0);
  Object* floor=new Rect(cement_floor, Point(-8, 8, 0),
			 Vector(4, 0, 0), Vector(0, 4, 0));
  ceiling_floor->add(floor);
  g->add(ceiling_floor);
  g->add(north_wall);
  g->add(west_wall);
  g->add(south_wall);
  g->add(east_wall);
}

SpinningInstance *make_dna(Group *g) {
  Phong *red = new Phong(Color(0.8,0.2,0.2), Color(0.5,0.5,0.5), 40, 0.4);
  Phong *yellow = new Phong(Color(0.8,0.8,0.2), Color(0.5,0.5,0.5), 40, 0.4);
  Phong *green = new Phong(Color(0.2,0.8,0.2), Color(0.5,0.5,0.5), 40, 0.4);
  Phong *blue = new Phong(Color(0.2,0.2,0.8), Color(0.5,0.5,0.5), 40, 0.4);
  Phong *white = new Phong(Color(0.3,0.3,0.3), Color(0.3,0.3,0.3), 20, 0);
  Point center_base(-8+3.7,8+3.7,0.63);
  Vector up(0,0,1);
  Vector left(1,0,0);
  Vector in(0,1,0);
  double rad=0.1;
  double dh=0.04;
  double th0=M_PI/4;
  double dth=M_PI/6;
  double srad=0.025;
  double crad=0.009;

  Group *g1 = new Group;
  for (int i=0; i<14; i++) {
    Phong *l, *r;
    if (drand48() < 0.5) { l=red; r=green; }
    else { l=yellow; r=blue; }
    Vector v(left*(rad*cos(th0+dth*i))+in*(rad*sin(th0+dth*i)));
    Point ls(center_base+up*(dh*i)+v);
    Point rs(center_base+up*(dh*i)-v);
    g1->add(new Sphere(l, ls, srad));
    g1->add(new Sphere(r, rs, srad));
    g1->add(new Cylinder(white, ls, rs, crad));
  }

  Grid *g2 = new Grid(g1, 5);
  Transform *mt = new Transform;
  InstanceWrapperObject *mw = new InstanceWrapperObject(g2);
  SpinningInstance *smw = new SpinningInstance(mw, mt, center_base, up, 0.2);
  g->add(smw);
  return smw;
}

void add_objects(Group *g, const Point &center) {
  Transform room_trans;
  room_trans.pre_translate(center.vector());

  string pathname("/usr/sci/data/Geometry/models/science-room/");

  Array1<int> sizes;
  Array1<string> names;

  names.add(string("386dx"));
  sizes.add(32);
  names.add(string("3d-glasses-01"));
  sizes.add(16);
  names.add(string("3d-glasses-02"));
  sizes.add(16);
  names.add(string("abacus"));
  sizes.add(32);
  names.add(string("coffee-cup-01"));
  sizes.add(16);
  names.add(string("coffee-cup-02"));
  sizes.add(16);
  names.add(string("coffee-cup-03"));
  sizes.add(16);
  names.add(string("coffee-cup-04"));
  sizes.add(16);
  names.add(string("coffee-cup-05"));
  sizes.add(16);
  names.add(string("coffee-cup-06"));
  sizes.add(16);
  names.add(string("coffee-cup-07"));
  sizes.add(16);
  names.add(string("coffee-cup-08"));
  sizes.add(16);
  names.add(string("corbusier-01"));
  sizes.add(16);
  names.add(string("corbusier-02"));
  sizes.add(16);
  names.add(string("corbusier-03"));
  sizes.add(16);
  names.add(string("corbusier-04"));
  sizes.add(16);
  names.add(string("corbusier-05"));
  sizes.add(16);
  names.add(string("end-table-01"));
  sizes.add(16);
  names.add(string("end-table-02"));
  sizes.add(16);
  names.add(string("end-table-03"));
  sizes.add(16);
  names.add(string("end-table-04"));
  sizes.add(16);
  names.add(string("end-table-05"));
  sizes.add(16);
  names.add(string("faucet-01"));
  sizes.add(32);
  names.add(string("sink-01"));
  sizes.add(16);
  names.add(string("futuristic-curio1"));
  sizes.add(16);
  names.add(string("futuristic-curio2"));
  sizes.add(16);
  names.add(string("hmd"));
  sizes.add(16);
  names.add(string("microscope"));
  sizes.add(16);
  names.add(string("plant-01"));
  sizes.add(32);

  Array1<Material *> matls;
  int i;
  for (i=0; i<names.size(); i++) {
    cerr << "Reading: "<<names[i]<<"\n";
    string objname(pathname+names[i]+string(".obj"));
    string mtlname(pathname+names[i]+string(".mtl"));
    if (!readObjFile(objname, mtlname, room_trans, g, sizes[i]))
      exit(0);
  }
}

extern "C"
Scene* make_scene(int argc, char* argv[], int nworkers)
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
  Group *g=new Group;
  make_walls_and_posters(g, center);

  Group* table=new Group();

  ImageMaterial *cement_pedestal = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/science-room/cement-pedestal.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp, 0,
		      Color(0,0,0), 0);
  table->add(new UVCylinder(cement_pedestal, center,
			    center+Vector(0,0,0.5), 1.5));
  Material *silver = new MetalMaterial(Color(0.7,0.73,0.8), 12);
  table->add(new Disc(silver, center+Vector(0,0,0.5),
		      Vector(0,0,1), 1.5));
  g->add(table);

  add_objects(g, center);

  SpinningInstance *smw = make_dna(g);

  //PER MATERIAL LIGHTS FOR THE HOLOGRAMS
  Light *holo_light0 = new Light(Point(-8, 8, 3.8), Color(0.3,0.3,0.4), 0);
  Light *holo_light1 = new Light(Point(-10, 10, 2.5), Color(0.3,0.3,0.4), 0);
  Light *holo_light2 = new Light(Point(-10, 6, 2.5), Color(0.3,0.3,0.4), 0);
  Light *holo_light3 = new Light(Point(-6, 6, 2.5), Color(0.3,0.3,0.4), 0);
  Light *holo_light4 = new Light(Point(-6, 10, 2.5), Color(0.3,0.3,0.4), 0);
  holo_light0->name_ = "hololight0";
  holo_light1->name_ = "hololight1";
  holo_light2->name_ = "hololight2";
  holo_light3->name_ = "hololight3";
  holo_light4->name_ = "hololight4";

  //CUTTING PLANE FOR THE HOLOGRAMS
  CutPlaneDpy* cpdpy=new CutPlaneDpy(Vector(.707,-.707,0), Point(-8,8,2));


  //ADD THE VISIBLE FEMALE DATASET
#ifdef ADD_VIS_FEM
  ColorMap *vcmap = new ColorMap("/usr/sci/data/Geometry/volumes/vfem",256);
  Material* vmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  vmat->my_lights.add(holo_light0);
  vmat->my_lights.add(holo_light1);
  vmat->my_lights.add(holo_light2);
  vmat->my_lights.add(holo_light3);
  vmat->my_lights.add(holo_light4);

  Material *vcutmat = new CutMaterial(vmat, vcmap, cpdpy);
  vcutmat->my_lights.add(holo_light0);
  vcutmat->my_lights.add(holo_light1);
  vcutmat->my_lights.add(holo_light2);
  vcutmat->my_lights.add(holo_light3);
  vcutmat->my_lights.add(holo_light4);

  CutVolumeDpy* vcvdpy = new CutVolumeDpy(1200.5, vcmap);
  
  HVolumeBrick16* slc0=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_0",
					  3, nworkers);
  
  HVolumeBrick16* slc1=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_1",
					  3, nworkers);
  
  HVolumeBrick16* slc2=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_2",
					  3, nworkers);

  HVolumeBrick16* slc3=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_3",
					  3, nworkers);
  
  HVolumeBrick16* slc4=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_4",
					  3, nworkers);
  
  HVolumeBrick16* slc5=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_5",
					  3, nworkers);

  HVolumeBrick16* slc6=new HVolumeBrick16(vcutmat, vcvdpy,
					  "/usr/sci/data/Geometry/volumes/vfem16_6",
					  3, nworkers);
					  

  Group *vig = new Group();
  vig->add(slc0);
  vig->add(slc1);
  vig->add(slc2);
  vig->add(slc3);
  vig->add(slc4);
  vig->add(slc5);
  vig->add(slc6);
  InstanceWrapperObject *viw = new InstanceWrapperObject(vig);

  Transform *vtrans = new Transform();
  vtrans->pre_translate(Vector(8, -8, -2));
  vtrans->pre_rotate(3.14/2.0, Vector(0,1,0));
  vtrans->pre_translate(Vector(-8, 8, 2));

  SpinningInstance *vinst = new SpinningInstance(viw, vtrans, Point(-8,8,2), Vector(0,0,1), 0.1);
  vinst->name_ = "Spinning Visible Woman";

  CutGroup *vcut = new CutGroup(cpdpy);
  vcut->add(vinst);
#endif

#ifdef ADD_DAVE_HEAD
  //ADD THE DAVE HEAD DATA SET
  ColorMap *hcmap = new ColorMap("/usr/sci/data/Geometry/volumes/dave",256);
  Material *hmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  hmat->my_lights.add(holo_light0);
  hmat->my_lights.add(holo_light1);
  hmat->my_lights.add(holo_light2);
  hmat->my_lights.add(holo_light3);
  hmat->my_lights.add(holo_light4);

  Material *hcutmat = new CutMaterial(hmat, hcmap, cpdpy);
  hcutmat->my_lights.add(holo_light0);
  hcutmat->my_lights.add(holo_light1);
  hcutmat->my_lights.add(holo_light2);
  hcutmat->my_lights.add(holo_light3);
  hcutmat->my_lights.add(holo_light4);

  CutVolumeDpy* hcvdpy = new CutVolumeDpy(82.5, hcmap);

  HVolumeBrick16* davehead=new HVolumeBrick16(hcutmat, hcvdpy,
					      "/usr/sci/data/Geometry/volumes/dave",
					      3, nworkers);
  InstanceWrapperObject *diw = new InstanceWrapperObject(davehead);

  Transform *dtrans = new Transform();
  dtrans->pre_translate(Vector(8, -8, -2)); 
  dtrans->rotate(Vector(1,0,0), Vector(0,0,1));
  dtrans->pre_translate(Vector(-8, 8, 2));

  SpinningInstance *dinst = new SpinningInstance(diw, dtrans, Point(-8,8,2), Vector(0,0,1), 0.1);
  dinst->name_ = "Spinning Head";

  CutGroup *hcut = new CutGroup(cpdpy);
  hcut->add(dinst);
  hcut->name_ = "Cutting Plane";
#endif

#ifdef ADD_CSAFE_FIRE
  //ADD THE CSAFE HEPTAINE POOL FIRE DATA SET
  Material* fmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  fmat->my_lights.add(holo_light0);
  fmat->my_lights.add(holo_light1);
  fmat->my_lights.add(holo_light2);
  fmat->my_lights.add(holo_light3);

  VolumeDpy* firedpy = new VolumeDpy(1000);

  int fstart = 55;
  int fend = 56;
  int finc = 8;
  SelectableGroup *fire_time = new SelectableGroup(1);
  fire_time->name_ = "CSAFE Fire Time Step Selector";
  //  TimeObj *fire_time = new TimeObj(5);
  for(int f = fstart; f < fend; f+= finc) {
    char buf[1000];
    sprintf(buf, "/usr/sci/data/CSAFE/heptane300_3D_NRRD/float/h300_%04df.raw",
	    f);
    cout << "Reading "<<buf<<endl;
    Object *fire=new HVolume<float, BrickArray3<float>, BrickArray3<VMCell<float> > > (fmat, firedpy, buf, 3, nworkers);
    fire_time->add(fire);
  }
  
  InstanceWrapperObject *fire_iw = new InstanceWrapperObject(fire_time);

  Transform *fire_trans = new Transform();
  fire_trans->pre_translate(Vector(8, -8, -2)); 
  fire_trans->rotate(Vector(1,0,0), Vector(0,0,1));
  fire_trans->pre_translate(Vector(-8, 8, 2));

  SpinningInstance *fire_inst = new SpinningInstance(fire_iw, fire_trans, Point(-8,8,2), Vector(0,0,1), 0.5);
  fire_inst->name_ = "Spinning CSAFE Fire";

  CutGroup *fire_cut = new CutGroup(cpdpy);
  fire_cut->add(fire_inst);
#endif
  
#ifdef ADD_GEO_DATA
  //ADD THE GEOLOGY DATA SET
  ColorMap *gcmap = new ColorMap("/usr/sci/data/Geometry/volumes/Seismic/geo",256);
  Material* gmat=new LambertianMaterial(Color(0.7,0.7,0.7));
  gmat->my_lights.add(holo_light0);
  gmat->my_lights.add(holo_light1);
  gmat->my_lights.add(holo_light2);
  gmat->my_lights.add(holo_light3);
  gmat->my_lights.add(holo_light4);

  Material *gcutmat = new CutMaterial(gmat, gcmap, cpdpy);
  gcutmat->my_lights.add(holo_light0);
  gcutmat->my_lights.add(holo_light1);
  gcutmat->my_lights.add(holo_light2);
  gcutmat->my_lights.add(holo_light3);
  gcutmat->my_lights.add(holo_light4);

  CutVolumeDpy* gcvdpy = new CutVolumeDpy(16137.7, gcmap);

  HVolumeBrick16* geology=new HVolumeBrick16(gcutmat, gcvdpy,
					      "/usr/sci/data/Geometry/volumes/Seismic/stack-16full.raw",
					      3, nworkers);
  InstanceWrapperObject *giw = new InstanceWrapperObject(geology);

  Transform *gtrans = new Transform();
  gtrans->pre_translate(Vector(8, -8, -2)); 
  gtrans->rotate(Vector(1,0,0), Vector(0,0,1));
  gtrans->pre_translate(Vector(-8, 8, 2));

  SpinningInstance *ginst = new SpinningInstance(giw, gtrans, Point(-8,8,2), Vector(0,0,1), 0.1);
  ginst->name_ = "Spinning Geology";

  CutGroup *gcut = new CutGroup(cpdpy);
  gcut->name_ = "Geology Cutting Plane";
  gcut->add(ginst);
#endif

  //PUT THE VOLUMES INTO A SWITCHING GROUP  
  SelectableGroup *sg = new SelectableGroup(60);
#ifdef ADD_VIS_FEM
  sg->add(vcut);
#endif
#ifdef ADD_DAVE_HEAD
  sg->add(hcut);
#endif
#ifdef ADD_CSAFE_FIRE
  sg->add(fire_inst);
#endif
#ifdef ADD_GEO_DATA
  sg->add(gcut);
#endif
  sg->name_ = "VolVis Selection";

  g->add(sg);

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.3, 0.3, 0.3);

  Scene *scene = new Scene(new Grid(g, 16),
			   cam, bgcolor, cdown, cup, groundplane, 0.3);
//  Scene *scene = new Scene(new HierarchicalGrid(g, 8, 8, 8, 20, 20, 5),
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);
//  Scene *scene = new Scene(g,
//			   cam, bgcolor, cdown, cup, groundplane, 0.3);

  scene->addObjectOfInterest( smw, true);  

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;
  Light *science_room_light0 = new Light(Point(-8, 8, 3.9), Color(.5,.5,.5), 0);
  science_room_light0->name_ = "science room overhead";
  scene->add_light(science_room_light0);
  Light *science_room_light1 = new Light(Point(-11, 11, 3), Color(.5,.5,.5), 0);
  science_room_light1->name_ = "science room corner";
  scene->add_light(science_room_light1);
  scene->animate=true;

  scene->addObjectOfInterest( sg, true );
#ifdef ADD_VIS_FEM
  scene->addObjectOfInterest( vinst, false );
  scene->attach_auxiliary_display(vcvdpy);
  vcvdpy->setName("Visible Female Volume");
  scene->attach_display(vcvdpy);
  (new Thread(vcvdpy, "VFEM Volume Dpy"))->detach();
#endif
#ifdef ADD_DAVE_HEAD
  scene->addObjectOfInterest( dinst, false );
  scene->attach_auxiliary_display(hcvdpy);
  hcvdpy->setName("Brain Volume");
  scene->attach_display(hcvdpy);
  (new Thread(hcvdpy, "HEAD Volume Dpy"))->detach();
#endif
#ifdef ADD_CSAFE_FIRE
  scene->addObjectOfInterest( fire_inst, true );
  scene->attach_auxiliary_display(firedpy);
  firedpy->setName("CSAFE Fire Volume");
  scene->attach_display(firedpy);
  (new Thread(firedpy, "CSAFE Fire Volume Dpy"))->detach();
#endif
#ifdef ADD_GEO_DATA
  scene->addObjectOfInterest( ginst, false );
  scene->attach_auxiliary_display(gcvdpy);
  gcvdpy->setName("Geological Volume");
  scene->attach_display(gcvdpy);
  (new Thread(gcvdpy, "GEO Volume Dpy"))->detach();
#endif
  scene->addObjectOfInterest( hcut, false );
  scene->attach_auxiliary_display(cpdpy);
  cpdpy->setName("Cutting Plane");
  scene->attach_display(cpdpy);
  (new Thread(cpdpy, "CutPlane Dpy"))->detach();

  scene->add_per_matl_mood_light(holo_light0);
  scene->add_per_matl_mood_light(holo_light1);
  scene->add_per_matl_mood_light(holo_light2);
  scene->add_per_matl_mood_light(holo_light3);
  scene->add_per_matl_mood_light(holo_light4);

  return scene;
}
