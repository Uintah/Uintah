/* look from above:

rtrt -np 8 -eye -18.9261 -22.7011 52.5255 -lookat -7.20746 -8.61347 -16.643 -up 0.490986 -0.866164 -0.0932288 -fov 40 -scene scenes/multi-scene 2 -scene scenes/graphics-museum -scene scenes/seaworld-tubes

look from hallway:
rtrt -np 8 -eye -5.85 -6.2 2 -lookat -8.16796 -16.517 2 -up 0 0 1 -fov 60 -scene scenes/graphics-museum 

looking at David:
rtrt -np 8 -eye -18.5048 -25.9155 1.39435 -lookat -14.7188 -16.1192 0.164304 -up 0 0 1 -fov 60  -scene scenes/graphics-museum 
*/

#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Disc.h>
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
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/ply.h>

using namespace rtrt;

#define MAXBUFSIZE 256
#define SCALE 950

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
    for (j = 0; j < nprops; j++)
      printf ("property %s\n", plist[j]->name);
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

  Camera cam(Eye,Lookat,Up,fov);

  Material* flat_white = new LambertianMaterial(Color(.8,.8,.8));
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
  Material* marble3=new CrowMarble(1.0, 
				   Vector(2,1,0),
				   Color(0.5,0.6,0.6),
				   Color(0.4,0.55,0.52),
				   Color(0.35,0.45,0.42)
				   );
  Material* marble4=new CrowMarble(1.5, 
				   Vector(-1,3,0),
//  				   Color(0.4,0.3,0.2),
				   Color(0,0,0),
				   Color(0.35,0.34,0.32),
				   Color(0.20,0.24,0.24)
				   );
  Material* marble=new Checker(marble1,
			       marble2,
			       Vector(3,0,0), Vector(0,3,0));
  Object* check_floor=new Rect(marble, Point(-12, -16, 0),
			       Vector(8, 0, 0), Vector(0, 12, 0));
  Group* south_wall=new Group();
  Group* west_wall=new Group();
  Group* north_wall=new Group();
  Group* east_wall=new Group();
  Group* ceiling_floor=new Group();
  Group* partitions=new Group();
  Group *baseg = new Group();

  ceiling_floor->add(check_floor);

  /*
  Material* whittedimg = 
    new ImageMaterial(1, "/usr/sci/data/Geometry/textures/whitted",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);
  
  Object* pic1=
    new Parallelogram(whittedimg, Point(-7.35, -11.9, 2.5), 
		      Vector(0,0,-1), Vector(-1.3,0,0));

  Material* bumpimg = 
    new ImageMaterial(1, "/usr/sci/data/Geometry/textures/bump",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      Color(0,0,0), 1, Color(0,0,0), 0);

  Object* pic2=
    new Parallelogram(bumpimg, Point(-11.9, -8.65, 2.5), 
		      Vector(0, 0, -1), Vector(0, 1.3, 0));
  */

  Material* white = new LambertianMaterial(Color(0.8,0.8,0.8));

  south_wall->add(new Rect(white, Point(-12, -28, 4), 
		       Vector(8, 0, 0), Vector(0, 0, 4)));

  west_wall->add(new Rect(white, Point(-20, -16, 4), 
		       Vector(0, 12, 0), Vector(0, 0, 4)));

  //  north_wall->add(new Rect(white, Point(-12, -4, 4), 
  //		       Vector(8, 0, 0), Vector(0, 0, 4)));
  // doorway cut out of North wall for W. tube: attaches to Hologram scene

  north_wall->add(new Rect(white, Point(-15.5, -4, 4), 
		       Vector(4.5, 0, 0), Vector(0, 0, 4)));
  north_wall->add(new Rect(white, Point(-7.5, -4, 5), 
		       Vector(3.5, 0, 0), Vector(0, 0, 3)));
  north_wall->add(new Rect(white, Point(-6.5, -4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

  //  east_wall->add(new Rect(white, Point(-4, -16, 4), 
  //			  Vector(0, 12, 0), Vector(0, 0, 4)));

  // doorway cut out of East wall for S. tube: attaches to Sphere Room scene

  east_wall->add(new Rect(white, Point(-4, -17.5, 4), 
		       Vector(0, 10.5, 0), Vector(0, 0, 4)));
  east_wall->add(new Rect(white, Point(-4, -6, 5), 
		       Vector(0, 1, 0), Vector(0, 0, 3)));
  east_wall->add(new Rect(white, Point(-4, -4.5, 4), 
		       Vector(0, 0.5, 0), Vector(0, 0, 4)));

  /*
  ceiling_floor->add(new Rect(white, Point(-12, -16, 8),
		       Vector(8, 0, 0), Vector(0, 12, 0)));
  */

  partitions->add(new Box(white, Point(-8-.1,-24,0),
			  Point(-8+.1,-4,5)));

  partitions->add(new Box(white, Point(-16,-24-.1,0),
			  Point(-8,-24+.1,5)));

  partitions->add(new Box(white, Point(-20,-16-.1,0),
			  Point(-12,-16+.1,5)));

  // david pedestal

  Point dav_ped_top(-14,-20,1);
  
  baseg->add(new Cylinder(flat_white,
			  Point(-14,-20,0),
			  dav_ped_top,2.7/2.));
  baseg->add(new Disc(flat_white,dav_ped_top,Vector(0,0,1),2.7/2.));

  Group* davidg;

#if 0
  Transform david_pedT;

  david_pedT.pre_translate(dav_ped_top.asVector());

  davidg = new Group();
  davidg->add(new Box(flat_white,
		      david_pedT.project(Point(-1.033567, -0.596689, 0.000000)),
		      david_pedT.project(Point(1.033567, 0.596689, 5.240253))));
  // david model or surrogate
#else 
  davidg = read_ply("/usr/sci/data/Geometry/Stanford_Sculptures/david_2mm.ply",flat_white);

  BBox david_bbox;

  davidg->compute_bounds(david_bbox,0);

  Point min = david_bbox.min();
  Point max = david_bbox.max();
  Vector diag = david_bbox.diagonal();

  printf("BBox: min: %lf %lf %lf max: %lf %lf %lf\nDimensions: %lf %lf %lf\n",
	 min.x(),min.y(), min.z(),
	 max.x(),max.y(), max.z(),
	 diag.x(),diag.y(),diag.z());

  Transform davidT;

  davidT.pre_translate(-Vector((max.x()+min.x())/2.,min.y(),(max.z()+min.z())/2.)); // center david over 0
  davidT.pre_rotate(M_PI_2,Vector(1,0,0));  // make z up
  davidT.pre_scale(Vector(.001,.001,.001)); // make units meters
  davidT.pre_translate(dav_ped_top.asVector());

  davidg->transform(davidT);

  david_bbox.reset();
  davidg->compute_bounds(david_bbox,0);

  min = david_bbox.min();
  max = david_bbox.max();
  diag = david_bbox.diagonal();

  printf("BBox: min: %lf %lf %lf max: %lf %lf %lf\nDimensions: %lf %lf %lf\n",
	 min.x(),min.y(), min.z(),
	 max.x(),max.y(), max.z(),
	 diag.x(),diag.y(),diag.z());
#endif

  baseg->add(davidg);

  // history hall pedestals
  baseg->add(new Box(flat_white,Point(-5.375,-9.25,0),Point(-4.625,-8.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-11.25,0),Point(-6.625,-10.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-13.25,0),Point(-4.625,-12.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-15.25,0),Point(-6.625,-14.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-17.25,0),Point(-4.625,-16.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-19.25,0),Point(-6.625,-18.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-21.25,0),Point(-4.625,-20.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-23.25,0),Point(-6.625,-22.5,1.4)));
  baseg->add(new Box(flat_white,Point(-5.375,-25.25,0),Point(-4.625,-24.5,1.4)));
  baseg->add(new Box(flat_white,Point(-7.375,-27.25,0),Point(-6.625,-26.5,1.4)));

  baseg->add(new Box(flat_white,Point(-9.37,-25.25,0),Point(-8.625,-24.5,1.4)));
  baseg->add(new Box(flat_white,Point(-11.375,-27.25,0),Point(-10.625,-26.5,1.4)));
  baseg->add(new Box(flat_white,Point(-13.375,-25.25,0),Point(-12.625,-24.5,1.4)));
  baseg->add(new Box(flat_white,Point(-15.375,-27.25,0),Point(-14.625,-26.5,1.4)));



  Group *g = new Group();
  /*
  west_wall->add(pic1);
  south_wall->add(pic2);
  */

//  g->add(new BV1(south_wall));
//  g->add(new BV1(west_wall));
//  g->add(new BV1(north_wall));
//  g->add(new BV1(north_wall));
//  g->add(new BV1(east_wall));

  g->add(ceiling_floor);
  g->add(south_wall);
  g->add(west_wall);
  g->add(north_wall);
  g->add(east_wall);
  g->add(partitions);
  g->add(baseg);
  g->add(davidg);
  
  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);

  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.1, 0.1, 0.6);
  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5);
  scene->ambient_hack = false;

  scene->select_shadow_mode("hard");
  scene->maxdepth = 8;
  scene->add_light(new Light(Point(-6, -16, 7.9), Color(.8,.8,.8), 0));
  scene->add_light(new Light(Point(-12, -26, 7.9), Color(.8,.8,.8), 0));
  scene->animate=false;
  return scene;
}
