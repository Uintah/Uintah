#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Transform.h>
#include <Packages/rtrt/Core/Group.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <Packages/rtrt/Core/TriMesh.h>
#include <Packages/rtrt/Core/GridTris.h>
#include <Packages/rtrt/Core/MeshedTri.h>
#include <Packages/rtrt/Core/ply.h>
#include <Packages/rtrt/Core/PLYReader.h>
using namespace rtrt;

#define SCALE 700
/* user's vertex and face definitions for a polygonal object */

typedef struct Vertex {
  float x,y,z;             /* the usual 3-space position of a vertex */
  unsigned char r, g, b;
} Vertex;

typedef struct Vertex2 {
  float x[3];             /* the usual 3-space position of a vertex */
  unsigned char color[3];
} Vertex2;

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
  {"diffuse_red", PLY_UCHAR, PLY_UCHAR, offsetof(Vertex,r), 0, 0, 0, 0},
  {"diffuse_green", PLY_UCHAR, PLY_UCHAR, offsetof(Vertex,g), 0, 0, 0, 0},
  {"diffuse_blue", PLY_UCHAR, PLY_UCHAR, offsetof(Vertex,b), 0, 0, 0, 0},
};

PlyProperty face_props[] = { /* list of property information for a vertex */
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(Face,verts),
   1, PLY_UCHAR, PLY_UCHAR, offsetof(Face,nverts)},
};

PlyProperty tristrip_props[] = { /*list of property information for a vertex*/
  {"vertex_indices", PLY_INT, PLY_INT, offsetof(TriStrip,verts),
   1, PLY_INT, PLY_INT, offsetof(TriStrip,nverts)},
};

void
rtrt::read_ply(char *fname, Material* matl, TriMesh* &tm, Group* &g)
{
  int i,j,k;
  PlyFile *ply;
  int nelems;
  char **elist;
  int file_type;
  float version;
  int nprops;
  int num_elems;
  PlyProperty **plist;
  char *elem_name;
  int num_comments;
  char **comments;
  int num_obj_info;
  char **obj_info;

  //int NVerts=0;
 
  if (tm == NULL) 
    tm = new TriMesh();
  if (g == NULL)
    g = new Group();

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
      
      /* set up for getting vertex elements */
      
      ply_get_property (ply, elem_name, &vert_props[0]);
      ply_get_property (ply, elem_name, &vert_props[1]);
      ply_get_property (ply, elem_name, &vert_props[2]);
      
      bool has_colors = false;
      int junk;

      PlyElement* ply_elem = find_element(ply,elem_name);

      if (find_property(ply_elem,"diffuse_red",&junk) != NULL) {
	has_colors = true;
	ply_get_property (ply, elem_name, &vert_props[3]);
	ply_get_property (ply, elem_name, &vert_props[4]);
	ply_get_property (ply, elem_name, &vert_props[5]);
      }
	
      /* grab all the vertex elements */
      for (j = 0; j < num_elems; j++) {
	
        /* grab and element from the file */
	Vertex vert;
        ply_get_element (ply, (void *) &vert);

	tm->verts.add(Point(vert.x, vert.y, vert.z));

	if (has_colors)
	  tm->colors.add(Color(vert.r/255.,vert.g/255.,vert.b/255.));
	
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
	  int p0 = face.verts[0];
	  int p1 = face.verts[k+1];
	  int p2 = face.verts[k+2];
	  
	  Point pt0 = tm->verts[p0];
	  Point pt1 = tm->verts[p1];
	  Point pt2 = tm->verts[p2];

	  Vector n = Cross(pt2-pt0,pt1-pt0);

	  // remove "bad" triangles
	  if (n.length() < 1.E-16)
	    continue;

	  n.normalize();

	  tm->norms.add(n);
	  int vn0;

	  vn0 = tm->norms.size()-1;
	  
	  if (tm->colors.size())
	    g->add(new MeshedColoredTri(tm,p0,p1,p2,vn0));
	  else
	    g->add(new MeshedTri(matl,tm,p0,p1,p2,vn0));
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

	  Point pt0 = tm->verts[idx0];
	  Point pt1 = tm->verts[idx1];
	  Point pt2 = tm->verts[idx2];
	  
	  Vector n = Cross(pt2-pt0,pt1-pt0);

	  // remove "bad" triangles
	  if (n.length() < 1.E-16)
	    continue;

	  n.normalize();
	  
	  tm->norms.add(n);
	  int vn0;

	  vn0 = tm->norms.size()-1;
	  
	  if (tm->colors.size())
	    g->add(new MeshedColoredTri(tm,idx0,idx1,idx2,vn0));
	  else
	    g->add(new MeshedTri(matl,tm,idx0,idx1,idx2,vn0));

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


}

void
rtrt::read_ply(char *fname, GridTris* gt)
{
  if(gt->isCached()){
    cerr << "Skipping read_ply for: " << fname << ", thinking that it is cached\n";
    return;
  } else {
    cerr << fname << " is not cached by gridtri, reading it\n";
  }
  int i,j,k;
  PlyFile *ply;
  int nelems;
  char **elist;
  int file_type;
  float version;
  int nprops;
  int num_elems;
  PlyProperty **plist;
  char *elem_name;
  int num_comments;
  char **comments;
  int num_obj_info;
  char **obj_info;

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
      
      //NVerts = num_elems;

      /* set up for getting vertex elements */
      
      ply_get_property (ply, elem_name, &vert_props[0]);
      ply_get_property (ply, elem_name, &vert_props[1]);
      ply_get_property (ply, elem_name, &vert_props[2]);
      
      int junk;

      PlyElement* ply_elem = find_element(ply,elem_name);

      if (find_property(ply_elem,"diffuse_red",&junk) != NULL) {
	ply_get_property (ply, elem_name, &vert_props[3]);
	ply_get_property (ply, elem_name, &vert_props[4]);
	ply_get_property (ply, elem_name, &vert_props[5]);
	gt->clearFallback();
      }
	
      /* grab all the vertex elements */
      for (j = 0; j < num_elems; j++) {
	
        /* grab and element from the file */
	Vertex2 vert;
        ply_get_element (ply, (void *) &vert);
	gt->addVertex(vert.x, vert.color);
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
        for (k = 0; k < face.nverts-2; k++) 
	  gt->addTri(face.verts[0], face.verts[k+1], face.verts[k+2]);
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

	  gt->addTri(idx0, idx1, idx2);
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


}

void
rtrt::read_ply(char *fname, GridTris* gt, Transform* t)
{
  if(gt->isCached()){
    cerr << "Skipping read_ply for: " << fname << ", thinking that it is cached\n";
    return;
  } else {
    cerr << fname << " is not cached by gridtri, reading it\n";
  }
  int i,j,k;
  PlyFile *ply;
  int nelems;
  char **elist;
  int file_type;
  float version;
  int nprops;
  int num_elems;
  PlyProperty **plist;
  char *elem_name;
  int num_comments;
  char **comments;
  int num_obj_info;
  char **obj_info;

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
      
      //NVerts = num_elems;

      /* set up for getting vertex elements */
      
      ply_get_property (ply, elem_name, &vert_props[0]);
      ply_get_property (ply, elem_name, &vert_props[1]);
      ply_get_property (ply, elem_name, &vert_props[2]);
      
      int junk;

      PlyElement* ply_elem = find_element(ply,elem_name);

      if (find_property(ply_elem,"diffuse_red",&junk) != NULL) {
	ply_get_property (ply, elem_name, &vert_props[3]);
	ply_get_property (ply, elem_name, &vert_props[4]);
	ply_get_property (ply, elem_name, &vert_props[5]);
	gt->clearFallback();
      }
	
      /* grab all the vertex elements */
      for (j = 0; j < num_elems; j++) {
	
        /* grab and element from the file */
	Vertex2 vert;
        ply_get_element (ply, (void *) &vert);
	Point p (vert.x[0],vert.x[1],vert.x[2]);
	t->project_inplace(p);
	vert.x[0]=p.x();
	vert.x[1]=p.y();
	vert.x[2]=p.z();
	gt->addVertex(vert.x, vert.color);
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
        for (k = 0; k < face.nverts-2; k++) 
	  gt->addTri(face.verts[0], face.verts[k+1], face.verts[k+2]);
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

	  gt->addTri(idx0, idx1, idx2);
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


}
