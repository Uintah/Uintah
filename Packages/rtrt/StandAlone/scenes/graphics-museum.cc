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
#include <Packages/rtrt/Core/UVCylinder.h>
#include <Packages/rtrt/Core/UVCylinderArc.h>
#include <Packages/rtrt/Core/Cylinder.h>
#include <Packages/rtrt/Core/ply.h>
#include <Packages/rtrt/Core/BBox.h>

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
  Material* glass= new DielectricMaterial(1.5, 1.0, 0.1, 400.0, 
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

  /* add cylinders */
  Material* grey = new PhongMaterial(Color(.5,.5,.5),1,0.3,100,true);
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

void add_glass_box (Group* obj_group, Point CornerA, Point CornerB,
		    Vector Right) {

}

void build_history_hall (Group* main_group, Scene *scene) {

  Material* flat_yellow = new LambertianMaterial(Color(.8,.8,.0));
  Material* flat_white = new LambertianMaterial(Color(.8,.8,.8));
  Material* flat_grey = new LambertianMaterial(Color(.4,.4,.4));
  Material* orange = new Phong(Color(.7,.4,.0),Color(.2,.2,.2),200);
  Material* yellow = new PhongMaterial(Color(.7,.7,.0),1,0.3,100,true);
  Material* white = new PhongMaterial(Color(.8,.8,.8),1,0.3,100,true);
  Material* black = new PhongMaterial(Color(.0,.0,.0),1,0.3,100,true);
  Material* purple = new PhongMaterial(Color(.7,.0,.7),1,0.3,100,true);
  Material* glass= new DielectricMaterial(1.5, 1.0, 0.04, 400.0, 
					  Color(.80, .93 , .87), 
					  Color(1,1,1), true, 0.001);
  Material* silver = new MetalMaterial( Color(0.8, 0.8, 0.8) );

  FILE *fp;
  char buf[MAXBUFSIZE];
  char *name;
  double x,y,z;
  int subdivlevel = 3;

  /* **************** history hall **************** */

  // history hall pedestals
  Group *historyg = new Group();
  const float img_size = 1.3;     
  float img_div = 0.15;
  const float img_ht = 3.5;
  const float ped_size = 0.75;
  float ped_div = (img_size - ped_size)/2.;
  const float ped_ht = 1.4;

  /* **************** image on North wall in history hall **************** */

  Vector NorthRight (img_size,0,0);
  Vector NorthDown (0,0,-img_size);
  Point NorthPoint (-8-img_div-img_size, -24.15-IMG_EPS, img_ht);
  Point PedPoint (0, -25.25, 0);

  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/gourardC-fill.ppm",
		      Point(-6.75, -4-IMG_EPS, img_ht), Vector(1.5,0,0),
		      Vector(0,0,-1.5),historyg);

  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/jelloC-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      historyg);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/herb1280C-fill.ppm",
		      NorthPoint, NorthRight, NorthDown, historyg);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/tmp/alpha1-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,historyg);
  PedPoint.x(NorthPoint.x()+ped_div);
  historyg->add(new Box(flat_white,PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));

  

  NorthPoint += Vector(-2*img_div-img_size, 0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/museumC-fill.ppm",
      
		      NorthPoint, NorthRight, NorthDown,historyg);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/tinC-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      historyg);

  //  cerr << "North Wall: " << NorthPoint << endl;
  
  /* **************** image on East wall in history hall **************** */
  img_div = 0.10;
  Vector EastRight (0,-img_size,0);
  Vector EastDown (0,0,-img_size);
  PedPoint = Point (-4.625, 0, ped_ht);
  
  Point EastPoint (-4-IMG_EPS, -7-img_div, img_ht);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/phongC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);
  PedPoint.y(EastPoint.y()-ped_div);
  historyg->add(new Box(flat_white,PedPoint-Vector(ped_size,ped_size,ped_ht),
			PedPoint));

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/eggC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/blinnC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);
  PedPoint.y(EastPoint.y()-ped_div);
  historyg->add(new Box(flat_white,PedPoint-Vector(ped_size,ped_size,ped_ht),
			PedPoint));
  Point BumpMapPoint (PedPoint-Vector(ped_size/2.,ped_size/2.,0));

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/rthetaC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/vasesC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/ringsC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);
  PedPoint.y(EastPoint.y()-ped_div);
  historyg->add(new Box(flat_white,PedPoint-Vector(ped_size,ped_size,ped_ht),
			PedPoint));
  Point RingsPoint (PedPoint-Vector(ped_size/2.,ped_size/2.,0)); 
			
  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/reyesC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/boxmontageC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);
  PedPoint.y(EastPoint.y()-ped_div);
  historyg->add(new Box(flat_white,PedPoint-Vector(ped_size,ped_size,ped_ht),
			PedPoint));

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/tmp/museum-4.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/tmp/museum-1.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);
  PedPoint.y(EastPoint.y()-ped_div);
  historyg->add(new Box(flat_white,PedPoint-Vector(ped_size,ped_size,ped_ht),
			PedPoint));

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/mapleC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/luxoC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/chessC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);
  PedPoint.y(EastPoint.y()-ped_div);
  historyg->add(new Box(flat_white,PedPoint-Vector(ped_size,ped_size,ped_ht),
			PedPoint));

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/dancersC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      historyg);

  //  cerr << "East Wall:  " << EastPoint-Vector(0,img_size,0) << endl;

  /* **************** image on West wall in history hall **************** */
  
  img_div = 0.06;
  Vector WestRight (0,img_size,0);
  Vector WestDown (0,0,-img_size);
  Point WestPoint (-7.85+IMG_EPS, -4-img_div-img_size, img_ht);

  PedPoint = Point (-7.375, 0, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/VWC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);
  PedPoint.y(WestPoint.y()+ped_div);
  historyg->add(new Box(flat_white, PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));
  Vector VWVector (PedPoint.vector()+Vector(ped_size/2.,ped_size/2.,ped_ht));

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/catmullC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);
  
  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/newellC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);
  PedPoint.y(WestPoint.y()+ped_div);
  historyg->add(new Box(flat_white, PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));
  
  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/tea-potC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);
  PedPoint.y(WestPoint.y()+ped_div);
  historyg->add(new Box(flat_white, PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));
  Vector TeapotVector (PedPoint.vector()+Vector(ped_size/2.,ped_size/2.,ped_ht));

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/maxC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/blurC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/tmp/recursive-rt-fill.ppm",
      WestPoint, WestRight, WestDown, 
		      historyg);
  PedPoint.y(WestPoint.y()+ped_div);
  historyg->add(new Box(flat_white, PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));
  Point RTPoint (PedPoint+Vector(ped_size/2., ped_size/2., ped_ht));

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/tmp/tron.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);
  PedPoint.y(WestPoint.y()+ped_div);
  historyg->add(new Box(flat_white, PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));
  Vector TronVector(PedPoint.vector()+Vector(ped_size/2.,ped_size/2.,ped_ht));
		      
  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/morphineC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);
  PedPoint.y(WestPoint.y()+ped_div);
  historyg->add(new Box(flat_white, PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));
  
  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/beeC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/ballsC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
 		      historyg);
  PedPoint.y(WestPoint.y()+ped_div);
  historyg->add(new Box(flat_white, PedPoint, 
			PedPoint+Vector(ped_size,ped_size,ped_ht)));
  Point BallsPoint (PedPoint+Vector(ped_size/2., ped_size/2., ped_ht));
  
  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/girlC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/kitchenC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/BdanceC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      historyg);

  //  cerr << "West Wall:  " << WestPoint << endl;

  WestPoint = Point (-20+IMG_EPS, -27, img_ht+.4);
  add_image_on_wall ("/usr/sci/data/Geometry/textures/museum/tmp/museum-7.ppm",
		      WestPoint, Vector(0,2,0), Vector(0,0,-2),
		      historyg);

  /* **************** image on South wall in history hall **************** */
  img_div = .07;
  Vector SouthRight (-img_size, 0, 0);
  Vector SouthDown (0,0,-img_size);
  Point SouthPoint (-4-img_div, -28+IMG_EPS, img_ht);
  
  PedPoint = Point (-0,-26.5,ped_ht);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/vermeerC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);		      

  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/dreamC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);
  
  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/factoryC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown,
		      historyg);
  
  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/tmp/museum-2.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);
  PedPoint.x(SouthPoint.x()-ped_div);
  historyg->add(new Box(flat_white,PedPoint-Vector(ped_size,ped_size,ped_ht),
			PedPoint));

  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/knickC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);
  
  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/painterfig2P-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);
  
  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/towerC-fill.ppm",
     SouthPoint, SouthRight, SouthDown, 
historyg);

  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/SpatchIllustrationC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);
  
  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/accC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);
  
  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/openglC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);

  SouthPoint -= Vector(2*img_div+img_size,0,0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/history/space_cookiesC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      historyg);

  //  cerr << "South Wall: " << SouthPoint-Vector(img_size, 0,0) << endl;

  /* **************** teapot **************** */
  Transform teapotT;

  teapotT.pre_rotate(M_PI_4,Vector(0,0,1));
  teapotT.pre_scale(Vector(0.004, 0.004, 0.004));
  teapotT.pre_translate(TeapotVector);
  
   fp = fopen("/usr/sci/data/Geometry/models/teapot.dat","r");
  
  while (fscanf(fp,"%s",buf) != EOF) {
    if (!strcasecmp(buf,"bezier")) {
      int numumesh, numvmesh, numcoords=3;
      Mesh *m;
      Bezier *b;
      Point p;
      
      fscanf(fp,"%s",buf);
      name = new char[strlen(buf)+1];
      strcpy(name,buf);
      
      fscanf(fp,"%d %d %d\n",&numumesh,&numvmesh,&numcoords);
          m = new Mesh(numumesh,numvmesh);
          for (int j=0; j<numumesh; j++) {
            for (int k=0; k<numvmesh; k++) {
              fscanf(fp,"%lf %lf %lf",&x,&y,&z);
              p = teapotT.project(Point(x,y,z));
              m->mesh[j][k] = p;
            }
          }
          b = new Bezier(silver,m);
          b->SubDivide(subdivlevel,.5);
          main_group->add(b->MakeBVH());
    }
  }
  fclose(fp);

  /* **************** car **************** */
  Material* vwmat=new Phong (Color(.6,.6,0),Color(.5,.5,.5),30);
  fp = fopen("/usr/sci/data/Geometry/models/vw.geom","r");
  if (!fp) {
    fprintf(stderr,"No such file!\n");
    exit(-1);
  }
  
  int vertex_count,polygon_count,edge_count;
  int numverts;
  Transform vwT;
  int num_pts, pi0, pi1, pi2;

  vwT.pre_scale(Vector(0.005,0.005,0.005));
  vwT.pre_rotate(M_PI_2,Vector(0,1,0));
  vwT.pre_rotate(M_PI_2,Vector(1,0,0));
  //  vwT.pre_translate(Vector(-7,-10.83,ped_ht));
  //  vwT.pre_translate(Vector(-5,-8.83,ped_ht));
  vwT.pre_translate(VWVector);

  fscanf(fp,"%d %d %d\n",&vertex_count,&polygon_count,&edge_count);
  
  double (*vert)[3] = new double[vertex_count][3];

  for (int i=0; i<vertex_count; i++) 
      fscanf(fp,"%lf %lf %lf",&vert[i][0],&vert[i][1],&vert[i][2]);
  Group* vw=new Group();
  while(fscanf(fp,"%d %d %d %d",&numverts, &pi0, &pi1, &pi2) != EOF) 
  {
      
      vw->add(new Tri(vwmat,
                     vwT.project(Point(vert[pi0-1][0],vert[pi0-1][1],vert[pi0-1][2])),
                     vwT.project(Point(vert[pi1-1][0],vert[pi1-1][1],vert[pi1-1][2])),
                     vwT.project(Point(vert[pi2-1][0],vert[pi2-1][1],vert[pi2-1][2]))));
      
      for (int i=0; i<numverts-3; i++) 
      {
          pi1 = pi2;
          fscanf(fp,"%d",&pi2);
	  Tri* t;
	  if(pi0 != pi2 && pi1 != pi2 && pi0 != pi1){
	      vw->add((t=new Tri(vwmat,
				 vwT.project(Point(vert[pi0-1][0],vert[pi0-1][1],vert[pi0-1][2])),
				 vwT.project(Point(vert[pi1-1][0],vert[pi1-1][1],vert[pi1-1][2])),
				 vwT.project(Point(vert[pi2-1][0],vert[pi2-1][1],vert[pi2-1][2])))));
	      
	      if(t->isbad()){
		  cerr << "BAD: " << pi0 << ", " << pi1 << ", " << pi2 << '\n';
	      }
	  }
      }
  }
  main_group->add(vw);
  /* **************** bump-mapped sphere **************** */
  historyg->add (new Sphere(orange, BumpMapPoint+Vector(0,0,0.3),0.2));

  /* **************** ray-traced scene **************** */
  historyg->add (new Sphere(glass, RTPoint+Vector(0.25,-0.1,0.3),0.1));
  historyg->add (new Sphere(silver, RTPoint+Vector(0,0.1,0.2),0.08));
  /* -eye -5.43536 -13.0406 2 -lookat -15.9956 -12.5085 2 -up 0 0 1 -fov 60*/

  /* **************** Saturn scene **************** */
  historyg->add (new Sphere(flat_yellow, RingsPoint+Vector(0,0,0.3),0.15));
  historyg->add (new Ring(flat_grey, RingsPoint+Vector(0,0,0.3),
			  Vector(-.2,-.25,1),0.18,0.03));  
  historyg->add (new Ring(flat_white, RingsPoint+Vector(0,0,0.3),
			  Vector(-.2,-.25,1),0.2105,0.07));  
  historyg->add (new Ring(flat_grey, RingsPoint+Vector(0,0,0.3),
			  Vector(-.2,-.25,1),0.281,0.01));  
  historyg->add (new Ring(flat_white, RingsPoint+Vector(0,0,0.3),
			  Vector(-.2,-.25,1),0.2915,0.04));  

  /* **************** Tron Light Cycle **************** */
  Transform tron_trans;

  // first, get it centered at the origin (in x and y), and scale it
  tron_trans.pre_translate(Vector(0,-.5,0));
  tron_trans.pre_scale(Vector(0.22, 0.22, 0.22));

  // now rotate/translate it to the right angle/position
  Transform t = tron_trans;
  double rot=(M_PI/2.);
  t.pre_rotate(rot, Vector(1,0,0));
  t.pre_rotate(rot, Vector(0,0,1));
  t.pre_translate(TronVector+Vector(0,0,0.3));
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/LightSycle.obj",
		   "/usr/sci/data/Geometry/models/museum/LightSycle.mtl",
		   t, main_group)) {
      exit(0);
  }
  /* **************** Billiard Balls **************** */
  historyg->add (new Sphere(yellow, BallsPoint+Vector(-0.1,-0.2,0.2),0.07));
  historyg->add (new Sphere(yellow, BallsPoint+Vector(0,-0.03,0.2),0.07));
  historyg->add (new Sphere(white, BallsPoint+Vector(0.23,-0.05,0.2),0.07));
  historyg->add (new Sphere(black, BallsPoint+Vector(-0.1,0.18,0.2),0.07));
  historyg->add (new Sphere(purple, BallsPoint+Vector(-0.15,0.29,0.2),0.07));
  
  main_group->add(historyg);

  /* history hall global lights */
  //  scene->add_light(new Light(Point(-6, -16, 5), Color(.401,.4,.4), 0));
  Light *l;
 
  l = new Light(Point(-12, -26, 4), Color(.402,.4,.4), 0);
  l->name_ = "History Hall A";
  scene->add_light(l);
  scene->add_light(new Light(Point(-6, -10, 4), Color(.403,.4,.4), 0));

  //  g->add(new Sphere(flat_yellow,Point(-6,-16,5),0.5));
  //  g->add(new Sphere(flat_yellow,Point(-12,-26,5),0.5));
}

void build_david_room (Group* main_group, Scene *scene) {
  Material* david_white = new LambertianMaterial(Color(.8,.75,.7));
  Material* flat_white = new LambertianMaterial(Color(.8,.8,.8));
  Material* light_marble1 
    = new CrowMarble(4.5, Vector(.3, .3, 0), Color(.9,.9,.9), 
		     Color(.8, .8,.8), Color(.7, .7, .7)); 

  /* **************** david pedestal **************** */

  Group* davidp = new Group();
  Point dav_ped_top(-14,-20,1);
  
  davidp->add(new UVCylinder(light_marble1,
			  Point(-14,-20,0),
			  dav_ped_top,2.7/2.));
  davidp->add(new Disc(flat_white,dav_ped_top,Vector(0,0,1),2.7/2.));

  /* **************** David **************** */

#if 0
  Transform bender_trans;
  Point bender_center (-12.5,-20,0);

  // first, get it centered at the origin (in x and y), and scale it
  bender_trans.pre_translate(Vector(0,0,32));
  bender_trans.pre_scale(Vector(0.035, 0.035, 0.035));

    Transform bt(bender_trans);
    bt.pre_translate(Vector(-14,-20,1));
    if (!readObjFile("/usr/sci/data/Geometry/models/museum/bender-2.obj",
		     "/usr/sci/data/Geometry/models/museum/bender-2.mtl",
		     bt, main_group)) {
      exit(0);
    }

#else
    /*  
  Transform david_pedT;

  david_pedT.pre_translate(dav_ped_top.asVector());

  Group* davidg = new Group();
  davidg->add(new Box(david_white,
		      david_pedT.project(Point(-1.033567, -0.596689, 0.000000)),
		      david_pedT.project(Point(1.033567, 0.596689, 5.240253))));
  // david model or surrogate
  g->add(davidg);
    */

  Group* davidg = read_ply("/usr/sci/data/Geometry/Stanford_Sculptures/david_2mm.ply",david_white);

  BBox david_bbox;

  davidg->compute_bounds(david_bbox,0);

  Point min = david_bbox.min();
  Point max = david_bbox.max();
  Vector diag = david_bbox.diagonal();
  /*
  printf("BBox: min: %lf %lf %lf max: %lf %lf %lf\nDimensions: %lf %lf %lf\n",
	 min.x(),min.y(), min.z(),
	 max.x(),max.y(), max.z(),
	 diag.x(),diag.y(),diag.z());
  */
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
  main_group->add(davidg);

#endif

  /* **************** couches in David room **************** */
  Transform sofa_trans;
  Point sofa_center (-19,-20,0);

  // first, get it centered at the origin (in x and y), and scale it
  sofa_trans.pre_translate(Vector(0,-445,0));
  sofa_trans.pre_scale(Vector(0.001, 0.001, 0.002));

  // now rotate/translate it to the right angle/position
  for (int i=0; i<2; i++) {
    Transform t(sofa_trans);
    double rad=(M_PI/2.);
    t.pre_rotate(rad, Vector(0,0,1));
    t.pre_translate(sofa_center.vector()+Vector(10*i,0,0));
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/museum-couch.obj",
		   "/usr/sci/data/Geometry/models/museum/museum-couch.mtl",
		   t, main_group)) {
      exit(0);
    }
  }

  /* **************** rope barrier in David room **************** */
  Transform rope_trans;
  Point rope_center (-12.5,-20,0);

  // first, get it centered at the origin (in x and y), and scale it
  rope_trans.pre_translate(Vector(-23,0,0));
  rope_trans.pre_scale(Vector(0.022, 0.022, 0.04));

  // now rotate/translate it to the right angle/position
  double rot;
  Vector more_rope_trans;
  
  more_rope_trans = Vector (-14,-18.5,0);
  Transform t = rope_trans;
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }
  more_rope_trans = Vector (-14,-21.5,0);
  t = rope_trans;
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }
  rot = M_PI/2.;
  more_rope_trans = Vector (-12.5,-20,0);
  t = rope_trans;
  t.pre_rotate(rot, Vector(0,0,1));
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }
  more_rope_trans = Vector (-15.5,-20,0);
  t = rope_trans;
  t.pre_rotate(rot, Vector(0,0,1));
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }
  rot = M_PI/4.;
  more_rope_trans = Vector (-12.9,-21.1,0);
  t = rope_trans;
  t.pre_rotate(rot, Vector(0,0,1));
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }
  more_rope_trans = Vector (-15.1,-18.9,0);
  t = rope_trans;
  t.pre_rotate(rot, Vector(0,0,1));
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }
  rot = 3*M_PI/4.;
  more_rope_trans = Vector (-12.9,-18.9,0);
  t = rope_trans;
  t.pre_rotate(rot, Vector(0,0,1));
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }
  more_rope_trans = Vector (-15.1,-21.1,0);
  t = rope_trans;
  t.pre_rotate(rot, Vector(0,0,1));
  t.pre_translate(more_rope_trans);
  if (!readObjFile("/usr/sci/data/Geometry/models/museum/barrier-2.obj",
		   "/usr/sci/data/Geometry/models/museum/barrier-2.mtl",
		   t, main_group)) {
    exit(0);
  }


  /* **************** images on North partition in David room **************** */
  Group* david_nwall=new Group();
  Material* d_b1 = 
      new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-b1-fill.ppm",  
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_b1 =
    new Parallelogram(d_b1, Point(-18.5, -16.15-IMG_EPS, 3.0), 
		      Vector(-1.0,0,0), Vector(0,0,-1.0));
  Material* d_b2 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-b2-fill.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_b2 =
    new Parallelogram(d_b2, Point(-18.5+1.5, -16.15-IMG_EPS, 3.0),
		      Vector(-1.0,0,0), Vector(0,0,-1.0));
  Material* d_b3 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-b3-fill.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_b3 =
    new Parallelogram(d_b3, Point(-18.5+(1.5*2), -16.15-IMG_EPS, 3.0), 
		      Vector(-1.0,0,0), Vector(0,0,-1.0));
  Material* d_b4 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-b4-fill.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_b4 =
    new Parallelogram(d_b4, Point(-18.5+(1.5*3), -16.15-IMG_EPS, 3.0), 
		      Vector(-1.0,0,0), Vector(0,0,-1.0));
  Material* d_b5 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-b5-fill.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_b5 =
    new Parallelogram(d_b5, Point(-18.5+(1.5*4), -16.15-IMG_EPS, 3.0), 
		      Vector(-1.0,0,0), Vector(0,0,-1.0));
  Material* d_bs1 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/museum-paragraph-02-flop.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_bs1 = 
    new Parallelogram(d_bs1, Point(-18.5, -16.15-IMG_EPS, 1.85), 
		      Vector(-1.0,0,0), Vector(0,0,-0.5625));
  Object* david_bs2 = 
    new Parallelogram(d_bs1, Point(-18.5+1.5, -16.15-IMG_EPS, 1.85), 
		      Vector(-1.0,0,0), Vector(0,0,-0.5625));
  Object* david_bs3 = 
    new Parallelogram(d_bs1, Point(-18.5+(1.5*2), -16.15-IMG_EPS, 1.85), 
		      Vector(-1.0,0,0), Vector(0,0,-0.5625));
  Object* david_bs4 = 
    new Parallelogram(d_bs1, Point(-18.5+(1.5*3), -16.15-IMG_EPS, 1.85), 
		      Vector(-1.0,0,0), Vector(0,0,-0.5625));
  Object* david_bs5 = 
    new Parallelogram(d_bs1, Point(-18.5+(1.5*4), -16.15-IMG_EPS, 1.85), 
		      Vector(-1.0,0,0), Vector(0,0,-0.5625));

  david_nwall->add(david_b1);
  david_nwall->add(david_bs1);
  david_nwall->add(david_b2);
  david_nwall->add(david_bs2);
  david_nwall->add(david_b3);
  david_nwall->add(david_bs3);
  david_nwall->add(david_b4);
  david_nwall->add(david_bs4);
  david_nwall->add(david_b5);
  david_nwall->add(david_bs5);
  
  /* **************** images on South partition in David room **************** */
  Group* david_swall=new Group();
  Material* d_a2 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-a2-fill.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_a2 =
    new Parallelogram(d_a2, Point(-8.5, -23.85+IMG_EPS, 4.5), 
		      Vector(-2.0,0,0), Vector(0,0,-2.0));
  Material* d_c1 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-c1-fill.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_c1 =
    new Parallelogram(d_c1, Point(-8.5-2.5, -23.85+IMG_EPS, 4.5), 
		      Vector(-2.0,0,0), Vector(0,0,-2.0));
  Material* d_c2 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/david-c2-fill.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_c2 =
    new Parallelogram(d_c2, Point(-8.5-(2.5*2), -23.85+IMG_EPS, 4.5), 
		      Vector(-2.0,0,0), Vector(0,0,-2.0));

  Material* d_as1 = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/museum-paragraph-04.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_as1 = 
    new Parallelogram(d_as1, Point(-8.5, -23.85+IMG_EPS, 2.35), 
		      Vector(-2.0,0,0), Vector(0,0,-0.5625*2));

  Material* d_cs1a = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/museum-names-01.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);
  Object* david_cs1a = 
    new Parallelogram(d_cs1a, Point(-8.5-2.5, -23.85+IMG_EPS, 2.35), 
		      Vector(-0.95,0,0), Vector(0,0,-1.69));

  Material* d_cs1b = 
    new ImageMaterial("/usr/sci/data/Geometry/textures/david/museum-names-02.ppm",
		      ImageMaterial::Clamp, ImageMaterial::Clamp,
		      1, Color(0,0,0), 0);

  Object* david_cs1b = 
    new Parallelogram(d_cs1b, Point(-8.5-3.55, -23.85+IMG_EPS, 2.35), 
		      Vector(-0.95,0,0), Vector(0,0,-1.69));

  Object* david_cs2 = 
    new Parallelogram(d_as1, Point(-8.5-(2.5*2), -23.85+IMG_EPS, 2.35), 
		      Vector(-2.0,0,0), Vector(0,0,-0.5625*2));


  david_swall->add(david_a2);
  david_swall->add(david_as1);
  david_swall->add(david_c1);
  david_swall->add(david_cs1a);
  david_swall->add(david_cs1b);
  david_swall->add(david_c2);
  david_swall->add(david_cs2);

  main_group->add(david_nwall);
  main_group->add(david_swall);
  main_group->add(davidp);

  /* David room lights */
  /*
  scene->add_light(new Light(Point(-14, -18, 7.9), Color(.403,.4,.4), 0));
  scene->add_light(new Light(Point(-14, -22, 7.9), Color(.404,.4,.4), 0));
  */
  scene->add_light(new Light(Point(-14, -10, 7), Color(.405,.4,.4), 0));
  scene->add_light(new Light(Point(-11.3, -18.05, 4), Color(.406,.4,.4), 0));
  scene->add_light(new Light(Point(-17, -22, 1.4), Color(.407,.4,.4), 0));

  Light *l;
  l = (new Light(Point(-11,-22.25,7.9),Color (.4,.401,.4), 0));
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-17,-22.25,7.9),Color (.4,.402,.4), 0));
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-14.75,-20.75,1),Color (.4,.403,.4), 0));
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-17,-17.75,7.9),Color (.4,.404,.4), 0));
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-14,-16.15,1), Color (.4,.405,.4), 0));
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = new Light(Point(-14,-23.85,7.9),Color (.4,.406,.4), 0);
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-11,-17.75,7.9),	Color (.4,.407,.4), 0));
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-11.25,-20.75,3.5),Color (.4,.408,.4), 0)); 
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-15.2,-21,5),Color (.4,.409,.4), 0)); 
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  l = (new Light(Point(-13.8,-17.8,6),Color (.4,.41,.4), 0)); 
  scene->add_per_matl_light (l);
  david_white->my_lights.add (l);
  /*
  l = (new Light(Point(-11,-22.25,7.9),Color (.4,.4,.4), 0));
  scene->add_light (l);
  l = (new Light(Point(-17,-22.25,7.9),Color (.4,.4,.4), 0));
  scene->add_light (l);
  l = (new Light(Point(-14.75,-20.75,1),Color (.4,.4,.4), 0));
  scene->add_light (l);
  l = (new Light(Point(-17,-17.75,7.9),Color (.4,.4,.4), 0));
  scene->add_light (l);
  l = (new Light(Point(-14,-16.15,1), Color (.4,.4,.4), 0));
  scene->add_light (l);
  l = new Light(Point(-14,-23.85,7.9),Color (.4,.4,.4), 0);
  scene->add_light (l);
  l = (new Light(Point(-11,-17.75,7.9),	Color (.4,.4,.4), 0));
  scene->add_light (l);
  l = (new Light(Point(-11.25,-20.75,3.5),Color (.4,.4,.4), 0)); 
  scene->add_light (l);

  g->add(new Sphere(flat_yellow,Point(-11,-22.25,7.9),0.5));
  g->add(new Sphere(flat_yellow,Point(-17,-22.25,7.9),0.5));
  g->add(new Sphere(flat_yellow,Point(-14.75,-20.75,1),0.5));
  g->add(new Sphere(flat_yellow,Point(-17,-17.75,7.9),0.5));
  g->add(new Sphere(flat_yellow,Point(-14,-16.15,1), 0.5));
  g->add(new Sphere(flat_yellow,Point(-14,-23.85,7.9),0.5));
  g->add(new Sphere(flat_yellow,Point(-11,-17.75,7.9),0.5));
  g->add(new Sphere(flat_yellow,Point(-11.25,-20.75,3.5),0.5));
  */
}

  /* **************** modern graphics rooms **************** */

void build_modern_room (Group *main_group, Scene *scene) {
  Material* flat_white = new LambertianMaterial(Color(.8,.8,.8));
  Material* shinyred = new MetalMaterial( Color(0.8, 0.0, 0.08) );
  FILE *fp;

  //  pedestals
  Group* moderng = new Group();

  // along east wall
  moderng->add(new Box(flat_white,Point(-8.75-0.375,-8-0.375,0),
		       Point(-8.75+0.375,-8+0.375,1.4)));
  moderng->add(new Box(flat_white,Point(-8.75-0.375,-12-0.375,0),
		       Point(-8.75+0.375,-12+0.375,1.4)));

  // along south wall
  moderng->add(new Box(flat_white,Point(-14-0.375,-15-0.375,0),
		       Point(-14+0.375,-15+0.375,1.4)));
  moderng->add(new Box(flat_white,Point(-16-0.375,-15-0.375,0),
		       Point(-16+0.375,-15+0.375,1.4)));

  // along west wall 
  moderng->add(new Box(flat_white,Point(-18.5-0.375,-11-0.375,0),
		       Point(-18.5+0.375,-11+0.375,1.4)));
  moderng->add(new Box(flat_white,Point(-18.5-0.375,-7.5-0.375,0),
		       Point(-18.5+0.375,-7.5+0.375,1.4)));

  // along north wall
  moderng->add(new Box(flat_white,Point(-16.5-0.375,-6-0.375,0),
		       Point(-16.5+0.375,-6+0.375,1.4)));
  Point bun_ped_top (-14,-6,1.4);
  moderng->add(new Box(flat_white,Point(-14-0.375,-6-0.375,0),
		       Point(-14+0.375,-6+0.375,1.4)));
  // center of room
  moderng->add(new Box(flat_white,Point(-16-0.6,-12-0.6,0),
		       Point(-16+0.6,-12+0.6,0.9)));
  moderng->add(new Box(flat_white,Point(-12-0.6,-10-0.6,0),
		       Point(-12+0.6,-10+0.6,0.9)));

  // St Matthew's Pedestal in northwest corner of room
  /*
  UVCylinderArc* StMattPed = (new UVCylinderArc(light_marble1, Point (-20,-16,0),
						Point (-20,-16, 1), 1));
  //  StMattPed->set_arc (3.*M_PI_2/4.,M_PI_2);
  moderng->add(StMattPed);	

  */

  /* **************** image on North wall in modern room **************** */
  const float img_size = 1.3;     
  float img_div = 0.35;
  const float img_ht = 3.5;

  Vector NorthRight (img_size,0,0);
  Vector NorthDown (0,0,-img_size);
  Point NorthPoint (-8.1-img_div-img_size, -4.1-IMG_EPS, img_ht);

  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/Figure12C-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      moderng);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/aging-venusC-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      moderng);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/bookscroppedC-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      moderng);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/buddhasC-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      moderng);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/bugsC-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      moderng);

  NorthPoint += Vector(-2*img_div-img_size, 0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/bunnyC-fill.ppm",
		      NorthPoint, NorthRight, NorthDown,
		      moderng);

  //  cerr << "North Wall: " << NorthPoint << endl;

  Vector WestRight (0,img_size,0);
  Vector WestDown (0,0,-img_size);
  Point WestPoint (-20+IMG_EPS, -4-img_div-img_size, img_ht);

  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/chickenposter2C-fill.ppm",
		      WestPoint, WestRight, WestDown,
		      moderng);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/collage_summaryC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      moderng);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/discontinuityC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      moderng);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/discontinuityC-fill.ppm",
		      WestPoint, WestRight, WestDown, 
		      moderng);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/flower_combC-fill.ppm",
		      WestPoint, WestRight, WestDown, 		      
		      moderng);

  WestPoint -= Vector (0, 2*img_div+img_size, 0);
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/geriC-fill.ppm",
		      WestPoint, WestRight, WestDown, 		      
		      moderng);

  //  cerr << "West Wall:  " << WestPoint << endl;

  Vector SouthRight (-img_size, 0, 0);
  Vector SouthDown (0,0,-img_size);
  Point SouthPoint (-20+img_div+img_size, -15.85+IMG_EPS, img_ht);

  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/ir-imageC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      moderng);

  SouthPoint += Vector(2*img_div+img_size,0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/large-lakeC-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      moderng);

  SouthPoint += Vector(2*img_div+img_size,0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/louvreC-fill.ppm",

		      SouthPoint, SouthRight, SouthDown, 
		      moderng);

  SouthPoint += Vector(2*img_div+img_size,0,0);  
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/lumigraph2C-fill.ppm",
		      SouthPoint, SouthRight, SouthDown, 
		      moderng);

  //  cerr << "South Wall: " << SouthPoint-Vector(img_size, 0,0) << endl;

  Vector EastRight (0,-img_size,0);
  Vector EastDown (0,0,-img_size);
  Point EastPoint (-8.15-IMG_EPS, -4-img_div, img_ht);

  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/the_endC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      moderng);

  EastPoint -= Vector(0, 2*img_div+img_size, 0);   
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/subd-venusC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      moderng);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/storyC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      moderng);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/poolC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      moderng);

  EastPoint -= Vector(0, 2*img_div+img_size, 0); 
  add_poster_on_wall ("/usr/sci/data/Geometry/textures/museum/modern/mayaC-fill.ppm",
		      EastPoint, EastRight, EastDown, 
		      moderng);

  //  cerr << "East Wall:  " << EastPoint-Vector(0,img_size,0) << endl;

#if 0
  Transform bunnyT;
  fp = fopen("/usr/sci/data/Geometry/models/bun.ply","r");
  if (!fp) {
    fprintf(stderr,"No such file!\n");
    exit(-1);
  }
  int num_verts, num_tris;
  
  fscanf(fp,"%d %d",&num_verts,&num_tris);
  
  double (*vert)[3] = new double[num_verts][3];
  double conf,intensity;
  int i;
  Material *bunnymat = new Phong(Color(.63,.51,.5),Color(.3,.3,.3),400);
  
  bunnyT.pre_scale(Vector(SCALE,SCALE,SCALE));
  bunnyT.pre_translate(Vector( -14,-6.5,1.5));
  int num_pts, pi0, pi1, pi2;
  
  Group* bunny=new Group();
  for (i=0; i<num_tris; i++) {
    fscanf(fp,"%d %d %d %d\n",&num_pts,&pi0,&pi1,&pi2);
    bunny->add(new Tri(shinyred,
		       bunnyT.project(Point(vert[pi0][0],vert[pi0][1],vert[pi0][2])),
		       bunnyT.project(Point(vert[pi1][0],vert[pi1][1],vert[pi1][2])),
		       bunnyT.project(Point(vert[pi2][0],vert[pi2][1],vert[pi2][2]))));
  }
  delete vert;
  fclose(fp);

  main_group->add (bunny);
#endif

  main_group->add(moderng);

  /* modern room lights */
  scene->add_light(new Light(Point(-17, -7, 4), Color(.404,.4,.4), 0));
  //  scene->add_light(new Light(Point(-6, -16, 5), Color(.401,.4,.4), 0));
  //  scene->add_light(new Light(Point(-12, -26, 5), Color(.402,.4,.4), 0));

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

  Material* flat_white = new LambertianMaterial(Color(.8,.8,.8));
  Material* shinyred = new MetalMaterial( Color(0.8, 0.0, 0.08) );
  /*
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
  */
  Material* dark_marble1 
    = new CrowMarble(4.5, Vector(-.3, -.3, 0), Color(.05,.05, .05),
		     Color(.075, .075, .075), Color(.1, .1, .1));

  Material* marble=new Checker(dark_marble1,
			       dark_marble1,
			       Vector(3,0,0), Vector(0,3,0));

  Object* check_floor=new Rect(marble, Point(-12, -16, 0),
			       Vector(8, 0, 0), Vector(0, 12, 0));

  Group* south_wall=new Group();
  Group* west_wall=new Group();
  Group* north_wall=new Group();
  Group* east_wall=new Group();
  Group* ceiling_floor=new Group();
  Group* partitions=new Group();

  ceiling_floor->add(check_floor);

  Material* wall_white = new LambertianMaterial(Color(0.8,0.8,0.8));

  south_wall->add(new Rect(wall_white, Point(-12, -28, 4), 
		       Vector(8, 0, 0), Vector(0, 0, 4)));

  south_wall->add(new Rect(wall_white, Point(-12-.16, -28, 4), 
		       Vector(8, 0, 0), Vector(0, 0, 4)));

  west_wall->add(new Rect(wall_white, Point(-20, -16, 4), 
		       Vector(0, 12, 0), Vector(0, 0, 4)));

  west_wall->add(new Rect(wall_white, Point(-20-.16, -16, 4), 
		       Vector(0, 12, 0), Vector(0, 0, 4)));

  //  north_wall->add(new Rect(wall_white, Point(-12, -4, 4), 
  //		       Vector(8, 0, 0), Vector(0, 0, 4)));
  // doorway cut out of North wall for W. tube: attaches to Hologram scene

  north_wall->add(new Rect(wall_white, Point(-15.5, -4, 4), 
		       Vector(4.5, 0, 0), Vector(0, 0, 4)));
  north_wall->add(new Rect(wall_white, Point(-7.5, -4, 5), 
		       Vector(3.5, 0, 0), Vector(0, 0, 3)));
  north_wall->add(new Rect(wall_white, Point(-6.5, -4, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

  north_wall->add(new Rect(wall_white, Point(-15.5, -4+.16, 4), 
		       Vector(4.5, 0, 0), Vector(0, 0, 4)));
  north_wall->add(new Rect(wall_white, Point(-7.5, -4+.16, 5), 
		       Vector(3.5, 0, 0), Vector(0, 0, 3)));
  north_wall->add(new Rect(wall_white, Point(-6.5, -4+.16, 1), 
		       Vector(2.5, 0, 0), Vector(0, 0, 1)));

  //  east_wall->add(new Rect(wall_white, Point(-4, -16, 4), 
  //			  Vector(0, 12, 0), Vector(0, 0, 4)));

  // doorway cut out of East wall for S. tube: attaches to Sphere Room scene

  east_wall->add(new Rect(wall_white, Point(-4, -17.5, 4), 
		       Vector(0, 10.5, 0), Vector(0, 0, 4)));
  east_wall->add(new Rect(wall_white, Point(-4, -6, 5), 
		       Vector(0, 1, 0), Vector(0, 0, 3)));
  east_wall->add(new Rect(wall_white, Point(-4, -4.5, 4), 
		       Vector(0, 0.5, 0), Vector(0, 0, 4)));

  east_wall->add(new Rect(wall_white, Point(-4+.16, -17.5, 4), 
		       Vector(0, 10.5, 0), Vector(0, 0, 4)));
  east_wall->add(new Rect(wall_white, Point(-4+.16, -6, 5), 
		       Vector(0, 1, 0), Vector(0, 0, 3)));
  east_wall->add(new Rect(wall_white, Point(-4+.16, -4.5, 4), 
		       Vector(0, 0.5, 0), Vector(0, 0, 4)));

  ceiling_floor->add(new Rect(wall_white, Point(-12, -16, 8),
			      Vector(8.16, 0, 0), Vector(0, 12.16, 0)));

  partitions->add(new Box(wall_white, Point(-8-.15,-24,0),
			  Point(-8+.15,-4,5)));
  partitions->add(new Cylinder(wall_white, Point(-8,-4,5),
			       Point(-8,-24,5),0.15));


  partitions->add(new Box(wall_white, Point(-16,-24-.15,0),
			  Point(-8,-24+.15,5)));
  partitions->add(new Cylinder(wall_white, Point(-16,-24,0),
			       Point(-16,-24,5),0.15));
  partitions->add(new Cylinder(wall_white, Point(-16,-24,5),
			       Point(-8.1,-24,5),0.15));
  partitions->add(new Sphere(wall_white, Point(-16,-24,5),0.15));


  partitions->add(new Box(wall_white, Point(-20,-16-.15,0),
			  Point(-12,-16+.15,5)));
  partitions->add(new Cylinder(wall_white, Point(-12,-16,0),
			       Point(-12,-16,5),0.15));
  partitions->add(new Cylinder(wall_white, Point(-12,-16,5),
			       Point(-20,-16,5),0.15));
  partitions->add(new Sphere(wall_white, Point(-12,-16,5),0.15));

  Color cdown(0.1, 0.1, 0.1);
  Color cup(0.1, 0.1, 0.1);
  rtrt::Plane groundplane(Point(0,0,-5), Vector(0,0,1));
  Color bgcolor(0.1, 0.1, 0.6);

  //  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5); 

  Scene *scene = new Scene(g, cam, bgcolor, cdown, cup, groundplane, 0.5, 
			   Sphere_Ambient);
  EnvironmentMapBackground *emap = new EnvironmentMapBackground ("/usr/sci/data/Geometry/textures/holo-room/environmap2.ppm", Vector(0,0,1));
  scene->set_ambient_environment_map(emap);


  scene->select_shadow_mode( Hard_Shadows );
  scene->maxdepth = 8;
  
  build_david_room (g,scene);
  build_history_hall (g,scene);
  build_modern_room (g,scene);

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

