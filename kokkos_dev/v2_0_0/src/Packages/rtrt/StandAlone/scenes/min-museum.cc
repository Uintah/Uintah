/* look from above:

rtrt -np 8 -eye -18.9261 -22.7011 52.5255 -lookat -7.20746 -8.61347 -16.643 -up 0.490986 -0.866164 -0.0932288 -fov 40 -scene scenes/graphics-museum 

look from hallway:
./rtrt -np 15 -bv 4 -hgridcellsize 8 8 8 -eye -5.85 -6.2 2 -lookat -8.16796 -16.517 2 -up 0 0 1 -fov 60 -scene scenes/graphics-museum 

looking at David:
./rtrt -np 40 -eye -11.6982 -16.4997 1.42867 -lookat -12.18 -21.0565 1.42867 -up 0 0 1 -fov 66.9403 -scene scenes/graphics-museum

and

./rtrt -np 20 -bv 4 -hgridcellsize 8 8 8 -eye -19.0241 -27.0214 1.97122 -lookat -15.2381 -17.2251 1.97122 -up 0 0 1 -fov 60 -scene scenes/graphics-museum


BEHIND:
./rtrt -np 20 -bv 4 -hgridcellsize 8 8 8 -eye -8.66331 -14.4693 1.97982 -lookat -10.5978 -17.9173 1.97982 -up 0 0 1 -fov 66.9403 -scene scenes/graphics-museum


rtrt -np 8 -eye -18.5048 -25.9155 1.63435 -lookat -14.7188 -16.1192 0.164304 -up 0 0 1 -fov 60  -scene scenes/graphics-museum 

from the other dirction (at David):
rtrt -np 8 -eye -10.9222 -16.5818 1.630637 -lookat -11.404 -21.1386 0.630637 -up 0 0 1 -fov 66.9403 -scene scenes/graphics-museum

rtrt -np 14 -eye -10.2111 -16.2099 1.630637 -lookat -11.7826 -20.5142 0.630637 -up 0 0 1 -fov 66.9403 scene scenes/graphics-museum


*/

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
#include <Packages/rtrt/Core/Ring.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
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
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/ImageMaterial.h>
#include <Packages/rtrt/Core/Parallelogram.h>
#include <Packages/rtrt/Core/UVSphere.h>
#include <Packages/rtrt/Core/UVCylinder.h>
#include <Packages/rtrt/Core/UVCylinderArc.h>
#include <Packages/rtrt/Core/DiscArc.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/ply.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/PerlinBumpMaterial.h>

using namespace rtrt;
using namespace SCIRun;

#define MAXBUFSIZE 256
#define IMG_EPS 0.01
#define SCALE 500

typedef struct Vertex {
  float x,y,z;             /* the usual 3-space position of a vertex */
} Vertex;

typedef struct Face {
  unsigned char nverts;    /* number of vertex indices in list */
  int *verts;              /* vertex index list */
} Face;

typedef struct TriStrip {
  int nverts;    /* number of vertex indices in list */
  int *verts;              /* vertex index list */
} TriStrip;

char *elem_names[] = { /* list of the kinds of elements in the user's object */
  "vertex", "face", "tristrips"
};

PlyProperty vert_props[] = { /* list of property information for a vertex */
  {"x", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,x), 0, 0, 0, 0},
  {"y", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,y), 0, 0, 0, 0},
  {"z", PLY_FLOAT, PLY_FLOAT, offsetof(Vertex,z), 0, 0, 0, 0},
};

PlyProperty face_props[] = { /* list of property information for a vertex */
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
   1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
};

PlyProperty tristrip_props[] = { /*list of property information for a vertex*/
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(TriStrip,verts),
   1, PLY_INT, PLY_INT, offsetof(TriStrip,nverts)},
};

Group *
read_ply(char *fname, Material* matl)
{
  Group *g = new Group();
  int i,j,k;
  PlyFile *ply;
  int nelems;
  char **elist;
  int file_type;
  float version;
  int nprops;
  int num_elems;
  PlyProperty **plist;
  Vertex **vlist;
  char *elem_name;
  int num_comments;
  char **comments;
  int num_obj_info;
  char **obj_info;

  int NVerts=0;

  /* open a PLY file for reading */
  ply = ply_open_for_reading(fname, &nelems, &elist, &file_type, &version);
  /* print what we found out about the file */
  printf ("version %f\n", version);
  printf ("type %d\n", file_type);

  /* go through each kind of element that we learned is in the file */
  /* and read them */
  
  for (i = 0; i < nelems; i++) {

    /* get the description of the first element */
    elem_name = elist[i];
    plist = ply_get_element_description (ply, elem_name, &num_elems, &nprops);
    
    /* print the name of the element, for debugging */
    printf ("element %s %d\n", elem_name, num_elems);
    
    /* if we're on vertex elements, read them in */
    if (equal_strings ("vertex", elem_name)) {
      
      /* create a vertex list to hold all the vertices */
      vlist = (Vertex **) malloc (sizeof (Vertex *) * num_elems);
      
      NVerts = num_elems;

      /* set up for getting vertex elements */
      
      ply_get_property (ply, elem_name, &vert_props[0]);
      ply_get_property (ply, elem_name, &vert_props[1]);
      ply_get_property (ply, elem_name, &vert_props[2]);
      
      /* grab all the vertex elements */
      for (j = 0; j < num_elems; j++) {
	
        /* grab and element from the file */
        vlist[j] = (Vertex *) malloc (sizeof (Vertex));
        ply_get_element (ply, (void *) vlist[j]);
	
        /* print out vertex x,y,z for debugging */
//          printf ("vertex: %g %g %g\n", vlist[j]->x, vlist[j]->y, vlist[j]->z);
      }
    }
    
    /* if we're on face elements, read them in */
    if (equal_strings ("face", elem_name)) {
      
      /* set up for getting face elements */
      
      ply_get_property (ply, elem_name, &face_props[0]);
      
      /* grab all the face elements */

      // don't need to save faces for this application,
      // so discard after use.
      Face face;
      for (j = 0; j < num_elems; j++) {
	
        /* grab an element from the file */
        ply_get_element (ply, (void *) &face);
	
        /* print out face info, for debugging */
        for (k = 0; k < face.nverts-2; k++) {
	  Vertex *v0 = vlist[face.verts[0]];
	  Vertex *v1 = vlist[face.verts[k+1]];
	  Vertex *v2 = vlist[face.verts[k+2]];
	  
	  Point p0(v0->x,v0->y,v0->z);
	  Point p1(v1->x,v1->y,v1->z);
	  Point p2(v2->x,v2->y,v2->z);
	  
	  g->add(new Tri(matl,p0,p1,p2));
	}
      }
    }

    /* if we're on triangle strip elements, read them in */
    if (equal_strings ("tristrips", elem_name)) {
      
      /* set up for getting face elements */
      
      ply_get_property (ply, elem_name, &tristrip_props[0]);
      
      TriStrip strip;

      /* grab all the face elements */
      for (j = 0; j < num_elems; j++) {
	
        /* grab an element from the file */
        ply_get_element (ply, (void *) &strip);
	
	/* Make triangle strips from vertices */
        for (k = 0; k < strip.nverts-2; k++) {
	  int idx0 = strip.verts[k+0];
	  int idx1 = strip.verts[k+1];
	  int idx2 = strip.verts[k+2];

	  if (idx0 < 0 || idx1 < 0 || idx2 < 0)
	    {
	      continue;
	    }

	  Vertex *v0 = vlist[idx0];
	  Vertex *v1 = vlist[idx1];
	  Vertex *v2 = vlist[idx2];

	  Point p0(v0->x,v0->y,v0->z);
	  Point p1(v1->x,v1->y,v1->z);
	  Point p2(v2->x,v2->y,v2->z);

	  
	  g->add(new Tri(matl,p0,p1,p2));

	}
      }
    }
    /* print out the properties we got, for debugging */
    /*    for (j = 0; j < nprops; j++)
	  printf ("property %s\n", plist[j]->name); */
  }
  /* grab and print out the comments in the file */
  comments = ply_get_comments (ply, &num_comments);
  for (i = 0; i < num_comments; i++)
    printf ("comment = '%s'\n", comments[i]);
  
  /* grab and print out the object information */
  obj_info = ply_get_obj_info (ply, &num_obj_info);
  for (i = 0; i < num_obj_info; i++)
    printf ("obj_info = '%s'\n", obj_info[i]);

  /* close the PLY file */
  ply_close (ply);

  printf("*********************************DONE************************\n");

  for (int j=0; j<NVerts; j++)
      delete vlist[j];
  delete(vlist);

  return g;

}

void add_image_on_wall (char *image_name, const Point &top_left, 
			 const Vector &right, const Vector &down,
			 Group* wall_group) {
  Material* image_mat = 
    new ImageMaterial(image_name,ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0); 
  Object* image_obj = 
    new Parallelogram(image_mat, top_left, right, down);

  wall_group->add(image_obj);
}

void add_poster_on_wall (char *image_name, const Point &top_left, 
			 const Vector &right, const Vector &down,
			 Group* wall_group) {

  add_image_on_wall(image_name, top_left, right, down, wall_group);

  /* add glass frame */
  Material* glass= new DielectricMaterial(1.5, 1.0, 0.05, 400.0, 
					  Color(.80, .93 , .87), 
					  Color(1,1,1), false);

  Vector in = Cross (right,down);
  Vector out = Cross (down, right);
  in.normalize();
  out.normalize();
  in *= 0.01;
  out *= 0.05;

  BBox glass_bbox;
  glass_bbox.extend (top_left + in);
  glass_bbox.extend (top_left + out + right + down);
  
  wall_group->add(new Box (glass, glass_bbox.min(), glass_bbox.max()));
  //  wall_group->add(new Box (clear, glass_bbox.min(), glass_bbox.max()));

  /* add cylinders */
  Material* grey = new PhongMaterial(Color(.5,.5,.5),1,0.3,100);
  wall_group->add(new Cylinder (grey, top_left+in+right*0.05+down*0.05,
				top_left+out*1.1+right*0.05+down*0.05, 0.01));
  wall_group->add(new Disc (grey, top_left+out*1.1+right*0.05+down*0.05, 
			    out, 0.01));

  wall_group->add(new Cylinder (grey, top_left+in+right*0.95+down*0.05,
				top_left+out*1.1+right*0.95+down*0.05, 0.01));
  wall_group->add(new Disc (grey, top_left+out*1.1+right*0.95+down*0.05, 
			    out, 0.01));

  wall_group->add(new Cylinder (grey, top_left+in+right*0.05+down*0.95,
				top_left+out*1.1+right*0.05+down*0.95, 0.01));
  wall_group->add(new Disc (grey, top_left+out*1.1+right*0.05+down*0.95, 
			    out, 0.01));

  wall_group->add(new Cylinder (grey, top_left+in+right*0.95+down*0.95,
				top_left+out*1.1+right*0.95+down*0.95, 0.01));
  wall_group->add(new Disc (grey, top_left+out*1.1+right*0.95+down*0.95, 
			    out, 0.01));
}

extern "C"
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  for(int i=1;i<argc;i++) {
    cerr << "Unknown option: " << argv[i] << '\n';
    cerr << "Valid options for scene: " << argv[0] << '\n';
    return 0;
  }

  Point Eye(-5.85, -6.2, 2.0);
  Point Lookat(-13.5, -13.5, 2.0);
  Vector Up(0,0,1);
  double fov=60;
  Group *g = new Group();
 
  Camera cam(Eye,Lookat,Up,fov);

  /*
  Material* flat_white = new LambertianMaterial(Color(.8,.8,.8));
  Material* shinyred = new MetalMaterial( Color(0.8, 0.0, 0.08) );
  Material* marble1=new CrowMarble(5.0,
				   Vector(2,1,0),
				   Color(0.5,0.6,0.6),
				   Color(0.4,0.55,0.52),
				   Color(0.35,0.45,0.42));
  Material* marble2=new CrowMarble(7.5,
				   Vector(-1,3,0),
				   Color(0.4,0.3,0.2),
				   Color(0.35,0.34,0.32),
				   Color(0.20,0.24,0.24));
  Material* marble3=new CrowMarble(5.0, 
				   Vector(2,1,0),
				   Color(0.2,0.2,0.2),
				   Color(0,0,0),
				   Color(0.35,0.4,0.4)
				   );
  Material* marble4=new CrowMarble(7.5, 
				   Vector(-1,3,0),
				   Color(0,0,0),
				   Color(0.35,0.34,0.32),
				   Color(0.20,0.24,0.24)
				   );
				   
  Material* floor_mat = new LambertianMaterial(Color(.7,.7,.5));
  Material* dark_marble1 
    = new CrowMarble(4.5, Vector(-.3, -.3, 0), Color(.05,.05, .05),
		     Color(.075, .075, .075), Color(.1, .1, .1));

  Material* marble=new Checker(dark_marble1,
			       dark_marble1,
			       Vector(3,0,0), Vector(0,3,0));
  */

  Material* floor_mat = new ImageMaterial("/opt/SCIRun/data/Geometry/textures/museum/carpet/carpet_black_blued2.ppm",
					  ImageMaterial::Tile,
					  ImageMaterial::Tile, 1,
					  Color(0,0,0), 0);
  floor_mat->SetScale (10,10);

  Object* check_floor=new Rect(floor_mat, Point(-12, -16, 0),
			       Vector(8, 0, 0), Vector(0, 12, 0));

  Group* south_wall=new Group();
  Group* west_wall=new Group();
  Group* north_wall=new Group();
  Group* east_wall=new Group();
  Group* ceiling_floor=new Group();
  Group* partitions=new Group();

  ceiling_floor->add(check_floor);

  Material* wall_white = new ImageMaterial("/opt/SCIRun/data/Geometry/textures/museum/general/tex-wall.ppm",
					   ImageMaterial::Tile,
					   ImageMaterial::Tile, 1,
					   Color(0,0,0), 0);
  const float wall_width = .16;

  south_wall->add(new Rect(wall_white, Point(-12, -28, 4), 
		       Vector(8, 0, 0), Vector(0, 0, 4)));

  south_wall->add(new Rect(wall_white, Point(-12, -28-wall_width, 4), 
		       Vector(8+wall_width, 0, 0), Vector(0, 0, 4)));

  west_wall->add(new Rect(wall_white, Point(-20, -16, 4), 
		       Vector(0, 12, 0), Vector(0, 0, 4)));

  west_wall->add(new Rect(wall_white, Point(-20-wall_width, -16, 4), 
		       Vector(0, 12+wall_width, 0), Vector(0, 0, 4)));

  //  north_wall->add(new Rect(wall_white, Point(-12, -4, 4), 
  //		       Vector(8, 0, 0), Vector(0, 0, 4)));
  // doorway cut out of North wall for W. tube: attaches to Hologram scene
  // door is from (-9,-4,0) to (-11,-4,0)

  north_wall->add(new Rect(wall_white, Point(-15.5, -4, 4), 
		       Vector(4.5, 0, 0), Vector(0, 0, 4)));
  north_wall->add(new Rect(wall_white, Point(-7.5, -4, 5), 
		       Vector(3.5, 0, 0), Vector(0, 0, 3)));
  north_wall->add(new Rect(wall_white, Point(-6.5, -4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

  north_wall->add(new Rect(wall_white, Point(-15.5-wall_width/2., -4+wall_width, 4), 
		       Vector(4.5+wall_width/2., 0, 0), Vector(0, 0, 4)));
  north_wall->add(new Rect(wall_white, Point(-7.5+wall_width/2., -4+wall_width, 5), 
		       Vector(3.5+wall_width/2., 0, 0), Vector(0, 0, 3)));
  north_wall->add(new Rect(wall_white, Point(-6.5+wall_width/2., -4+wall_width, 1), 
		       Vector(2.5+wall_width/2., 0, 0), Vector(0, 0, 1)));

  //  east_wall->add(new Rect(wall_white, Point(-4, -16, 4), 
  //			  Vector(0, 12, 0), Vector(0, 0, 4)));

  // doorway cut out of East wall for S. tube: attaches to Sphere Room scene
  // door is from (-4,-5,0) to (-4,-7,0)

  east_wall->add(new Rect(wall_white, Point(-4, -17.5, 4), 
		       Vector(0, 10.5, 0), Vector(0, 0, 4)));
  east_wall->add(new Rect(wall_white, Point(-4, -6, 5), 
		       Vector(0, 1, 0), Vector(0, 0, 3)));
  east_wall->add(new Rect(wall_white, Point(-4, -4.5, 4), 
		       Vector(0, 0.5, 0), Vector(0, 0, 4)));

  east_wall->add(new Rect(wall_white, Point(-4+wall_width, -17.5, 4), 
		       Vector(0, 10.5+wall_width, 0), Vector(0, 0, 4)));
  east_wall->add(new Rect(wall_white, Point(-4+wall_width, -6+wall_width/2., 5), 
		       Vector(0, 1+wall_width/2., 0), Vector(0, 0, 3)));
  east_wall->add(new Rect(wall_white, Point(-4+wall_width, -4.5+wall_width/2., 4), 
		       Vector(0, 0.5+wall_width/2., 0), Vector(0, 0, 4)));

  //  ceiling_floor->add(new Rect(wall_white, Point(-12, -16, 8),
  //			      Vector(8.16, 0, 0), Vector(0, 12.16, 0)));

  partitions->add(new Rect(wall_white, Point(-8-.15,-14,2.5),
			   Vector(0,10,0),Vector(0,0,2.5)));
  partitions->add(new Rect(wall_white, Point(-8+.15,-14,2.5),
			   Vector(0,-10,0),Vector(0,0,2.5)));
  partitions->add(new UVCylinder(wall_white, Point(-8,-4,5),
			       Point(-8,-24-.15,5),0.15));
  partitions->add(new UVCylinder(wall_white, Point(-8,-24,0),
			       Point(-8,-24,5),0.15));
  partitions->add(new Sphere(wall_white, Point(-8,-24,5),0.15));


  partitions->add(new Rect(wall_white, Point(-12,-24-.15,2.5),
			   Vector(4, 0, 0), Vector(0,0,2.5)));
  partitions->add(new Rect(wall_white, Point(-12,-24+.15,2.5),
			   Vector(-4, 0, 0), Vector(0,0,2.5)));
  partitions->add(new UVCylinder(wall_white, Point(-16,-24,0),
			       Point(-16,-24,5),0.15));
  partitions->add(new UVCylinder(wall_white, Point(-16,-24,5),
			       Point(-8+.15,-24,5),0.15));
  partitions->add(new Sphere(wall_white, Point(-16,-24,5),0.15));


  partitions->add(new Rect(wall_white, Point(-16,-16-.15,2.5),
			   Vector(4,0,0), Vector(0,0,2.5)));
  partitions->add(new Rect(wall_white, Point(-16,-16+.15,2.5),
			   Vector(-4,0,0), Vector(0,0,2.5)));
  partitions->add(new UVCylinder(wall_white, Point(-12,-16,0),
			       Point(-12,-16,5),0.15));
  partitions->add(new UVCylinder(wall_white, Point(-12,-16,5),
			       Point(-20,-16,5),0.15));
  partitions->add(new Sphere(wall_white, Point(-12,-16,5),0.15));

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);
  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.1, 0.1, 0.6);

  //  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5); 

  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5, 
			   Sphere_Ambient);
  EnvironmentMapBackground *emap = new EnvironmentMapBackground ("/opt/SCIRun/data/Geometry/textures/holo-room/environmap2.ppm", Vector(0,0,1));
  scene->set_ambient_environment_map(emap);

  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;

  Transform outlet_trans;
  // first, get it centered at the origin (in x and y), and scale it
  outlet_trans.pre_translate(Vector(238,-9,-663));
  outlet_trans.pre_scale(Vector(0.00003, 0.00003, 0.00003));
  /*  
  // now rotate/translate it to the right angle/position
  Transform t = Transform (tron_trans);
  rot=(M_PI/2.);
  t.pre_rotate(rot, Vector(1,0,0));
  t.pre_translate(TronVector+Vector(10*i,0,0));
  */

  g->add(ceiling_floor);
  g->add(south_wall);
  g->add(west_wall);
  g->add(north_wall);
  g->add(east_wall);
  g->add(partitions);

  scene->animate=false;
  return scene;
}


