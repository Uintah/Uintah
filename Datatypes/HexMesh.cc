/*
 *  HexMesh.h: Unstructured meshes
 *
 *  Written by:
 *   Peter A. Jensen
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */


/*******************************************************************************
* Version control
*******************************************************************************/

#define HEXMESH_VERSION 1


/*******************************************************************************
* Includes
*******************************************************************************/

#include <Datatypes/HexMesh.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <fstream.h>

/*******************************************************************************
********************************************************************************
* Global variables and forward declarations.
********************************************************************************
*******************************************************************************/

static Persistent* make_HexMesh();

PersistentTypeID HexMesh::type_id("HexMesh", "Datatype", make_HexMesh);


/*******************************************************************************
********************************************************************************
* HexNode
********************************************************************************
*******************************************************************************/

/*******************************************************************************
* Constructor
*
* 	This constructor makes a point and assigns it an index.
*******************************************************************************/

HexNode::HexNode (int i, double x, double y, double z)
: Point (x, y, z),
  my_index (i)
{
}


/*******************************************************************************
* HexNode::HexNode ()
*
* 	The default constructor, it sets up all values of the node to 0.  This
* should only be used prior to doing a Pio call to read in this hex node.
*******************************************************************************/

HexNode::HexNode ()
: Point (0.0, 0.0, 0.0),
  my_index (0)
{
}


/*******************************************************************************
* ostream & operator << (ostream & o, const HexNode & n)
*
*	This function outputs the contents of the HexNode for debugging
* purposes only.
*******************************************************************************/

ostream & operator << (ostream & o, const HexNode & n)
{
  o << "HexNode " << (unsigned int) &n << ": x=" << n.x() << " y=" << n.y() 
    << " z=" << n.z() << " index=" << n.my_index
    << endl;
    
  return o;
}


/*******************************************************************************
* void Pio (Piostream & p, HexNode & n)
*
*	Output or input the node to the given io stream.
*******************************************************************************/

void Pio (Piostream & p, HexNode & n)
{
  // Write out or read in the data.
  
  p.begin_cheap_delim();
  
  Pio(p, n.my_index);
      
  // Write out or read in the parent class data.
  
  Pio (p, (Point &) n);
  
  p.end_cheap_delim();
}


/*******************************************************************************
* void Pio (Piostream & p, HexNode * & n)
*
*	Output or input the node to the given io stream.  Create a new node
* and return it if we are reading.
*******************************************************************************/

void Pio (Piostream & p, HexNode * & n)
{
  // If we are reading, then we must assume the node does not exist.
  //   Caution, this is prone to memory leaks.
  
  if (p.reading ())
    n = new HexNode ();
    
  // Use the other Pio routine for consistancy.
  
  Pio (p, *n); 
}


/*******************************************************************************
********************************************************************************
* HexFace
********************************************************************************
*******************************************************************************/

/*******************************************************************************
* HexFace::HexFace (int, int, FourHexNodes & f, HexMesh * m)
*
* 	This constructor makes a face and assigns nodes to it.  Note that all
* the nodes should have indecies and pointers (none should be NULL).
*******************************************************************************/

HexFace::HexFace (int i, int e, FourHexNodes & f, HexMesh * m)
: my_index (i), my_contains_index (e), my_neighbor_index (0)
{
  // Use the helper function to set up this face.

  set_corners (f);
  
  if (Dot (my_normal, m->find_element (my_contains_index)->centroid()) + my_d < 0)
  {
    my_normal *= -1.0;
  }
}


/*******************************************************************************
* HexFace::HexFace ()
*
* 	The default constructor, it sets up all values of the face to 0.  This
* should only be used prior to doing a Pio call to read in this hex face.
*******************************************************************************/

HexFace::HexFace ()
: my_index (0), my_contains_index (0), my_neighbor_index (0)
{
  int c;
  
  // Clear this face.
  
  for (c = 4; c--;)
  {
    corner.index[c] = 0;
    corner.node[c] = NULL;
  }
}


/*******************************************************************************
* void HexFace::calc_face ()
*
* 	This function sets up the centroid and the normal for this face.  It
* also calculates a measure for the planarity of the corners of the face.
*******************************************************************************/

void HexFace::calc_face ()
{
  Vector n;

  // Find the centroid.  (Uses 4 points, even if points are duplicated.)
  
  my_centroid = AffineCombination (*corner.node[0], 0.25, *corner.node[1], 0.25,
                                   *corner.node[2], 0.25, *corner.node[3], 0.25);

  // Find the normal by averaging the normals from each corner.  Make sure
  //   normals point in the same direction, add them, and then normalize.  
  //   Adding them first allows each corner to contribute proportionally to the
  //   contribution of that corner. 

  // Do all three pairs from each corner, 12 total operations.  Assuming that
  //   the corners are given counterclockwise, the normal faces the same
  //   direction as the corners are viewed.

  my_normal = Vector (0, 0, 0);

  my_normal += Cross (*corner.node[3] - *corner.node[0], 
                      *corner.node[1] - *corner.node[0]);
  my_normal += Cross (*corner.node[0] - *corner.node[1], 
                      *corner.node[2] - *corner.node[1]);
  my_normal += Cross (*corner.node[1] - *corner.node[2], 
                      *corner.node[3] - *corner.node[2]);
  my_normal += Cross (*corner.node[2] - *corner.node[3], 
                      *corner.node[0] - *corner.node[3]);
  
  // Normalize the normal.  If it's too short, assume this face is not a plane.
  
  if (my_normal.length() > 1.0e-10)
    my_normal.normalize ();
  else
    my_normal *= 0.0;
    
  // As a general rule, normals are reversed.
    
  my_normal *= -1.0;  
  
  // Find the distance between each corner and the plane.  (If the length of
  //   the normal is zero, no plane, this value is zero.)  Average them.
  
  planar_adherence  = fabs (Dot (my_normal, my_centroid - *corner.node[0]));
  planar_adherence += fabs (Dot (my_normal, my_centroid - *corner.node[1]));
  planar_adherence += fabs (Dot (my_normal, my_centroid - *corner.node[2]));
  planar_adherence += fabs (Dot (my_normal, my_centroid - *corner.node[3]));
  
  planar_adherence /= 4.0;
  
  // Set d.
  
  my_d = - Dot (my_normal, my_centroid);
}


/*******************************************************************************
* void HexFace::set_corners (FourHexNodes & f)
*
* 	This function re-initializes this face.
*******************************************************************************/

void HexFace::set_corners (FourHexNodes & f)
{
  // Make a copy of the corners.
  
  corner = f;

  // Recalculate the normal, centroid, and planar adherence.
  
  calc_face ();
}


/*******************************************************************************
* int HexFace::is_corner (int i)
*
* 	This function returns true if there is a corner in this face with
* index 'i'.
*******************************************************************************/

int HexFace::is_corner (int i)
{
  int c;
  
  for (c = 4; c--;)
    if (corner.index[c] == i)
      return 1;
      
  return 0;
}


/*******************************************************************************
* int HexFace::compare (FourHexNodes & f)
*
* 	This function returns true if the corners in this object match the
* corners passed as a parameter.  Only indecies are compared.
*******************************************************************************/

int HexFace::compare (FourHexNodes & f)
{
  int used[4] = { 0, 0, 0, 0 };
  int c, d, unmatched = 4;
  
  // Compare indecies.
  
  for (c = 4; c--;)
    for (d = 4; d--;)
      if (!used[d] && f.index[c] == corner.index[d])
      {
        unmatched--;
        used[d]++;
        break;
      }
      
  // Return truth of compare.
  
  return unmatched == 0;
}


/*******************************************************************************
* double HexFace::dist (const Point & P)
*
* 	This function returns the distance from the face of the given point.
* Note that if the plane is degenerate (i.e. a line or a point) then this
* returns 0.  Negative values indicate that the point lies in the opposite
* direction as the normal.
*******************************************************************************/

double HexFace::dist (const Point & P)
{
  return my_normal.x() * P.x() + my_normal.y() * P.y() + my_normal.z() * P.z ()
         + my_d;
}


/*******************************************************************************
* ostream & operator << (ostream & o, const HexFace & f)
*
*	This function outputs the contents of the HexFace for debugging
* purposes only.
*******************************************************************************/

ostream & operator << (ostream & o, const HexFace & f)
{
  o << "HexFace " << (unsigned int) &f << ": index=" << f.my_index
    << ": contains=" << f.my_contains_index << ": neighbor=" << f.my_neighbor_index
    << " centroid=" << f.my_centroid.x() << "," << f.my_centroid.y() 
    << "," << f.my_centroid.z() << " normal=" << f.my_normal.x() << "," 
    << f.my_normal.y() << "," << f.my_normal.z() 
    << endl;
    
  o << "    Nodes " << f.corner.index[0] << " " << f.corner.index[1] 
    << " " << f.corner.index[2] << " " << f.corner.index[3] 
    << endl;
    
  return o;
}


/*******************************************************************************
* void Pio (Piostream & p, HexFace & f)
*
*	Output or input the face to the given io stream.
*******************************************************************************/

void Pio (Piostream & p, HexFace & f)
{
  int c;
  
  // Write out or read in the data.
  
  p.begin_cheap_delim();
  
  Pio(p, f.my_index);
  Pio(p, f.my_contains_index);
  Pio(p, f.my_neighbor_index);
        
  for (c = 0; c < 4; c++)
    Pio (p, f.corner.index[c]);    
        
  p.end_cheap_delim();
}


/*******************************************************************************
* void Pio (Piostream & p, HexFace * & f)
*
*	Output or input the face to the given io stream.  Create a new face
* and return it if we are reading.
*******************************************************************************/

void Pio (Piostream & p, HexFace * & f)
{
  // If we are reading, then we must assume the face does not exist.
  //   Caution, this is prone to memory leaks.
  
  if (p.reading ())
    f = new HexFace ();
    
  // Use the other Pio routine for consistancy.
  
  Pio (p, *f); 
}


/*******************************************************************************
* void HexFace::finish_read (HexMesh * m)
*
*     Set up pointers which were not stored in permanent storage.
*******************************************************************************/

void HexFace::finish_read (HexMesh * m)
{
  int c;
  double d;
  
  for (c = 8; c--;)
    corner.node[c] = m->find_node (corner.index[c]);

  calc_face ();
  
  d = Dot (my_normal, m->find_element (my_contains_index)->centroid()) + my_d;
  if (d < 0)
    my_normal *= -1.0;  
}


/*******************************************************************************
********************************************************************************
* Hexahedron
********************************************************************************
*******************************************************************************/

/*******************************************************************************
* Hexahedron::Hexahedron (int index, HexMesh * m, EightHexNodes & e)
*
* 	Initialize the index and the vertecies of the volume.
*******************************************************************************/

Hexahedron::Hexahedron (int index, HexMesh * m, EightHexNodes & e)
: my_index (index),
  num_faces (6),
  corner (e),
  min (999999, 999999, 999999),
  max (0, 0, 0)
{
  static int face_order[] = { 0, 1, 2, 3,  0, 3, 7, 4,  0, 4, 5, 1,  
                              1, 5, 6, 2,  2, 6, 7, 3,  4, 7, 6, 5 };  
  int c, d, p;
  HexFace * t, * f;
  FourHexNodes n;
  
  calc_centroid ();
  
  for (p = 0, c = 0; c < 6; c++)
  {
    // Select nodes for face construction
  
    for (d = 0; d < 4; d++, p++)
    {
      n.index[d] = e.index[face_order[p]];
      n.node[d]  = e.node[face_order[p]];
    }
    
    // Is there already a face with these points?
    
    f = m->find_face (n);
    
    // Build a face
    
    face.index[c] = m->add_face (my_index, n);
    face.face[c] = m->find_face (face.index[c]);
    
    // Register neighbors
    
    if (f != NULL)
    {
      face.face[c]->set_neighbor_index (f->contains_index ());
      f->set_neighbor_index (my_index); 
    }    
  }  
  
  // Make sure the first faces listed are not degenerate.
  
  for (c = 0; c < num_faces; c++)
    if (face.face[c]->normal().length() <= 1e-10)
    {
      num_faces--;
      d = face.index[c];
      t = face.face[c];
      face.index[c] = face.index[num_faces];
      face.face[c] = face.face[num_faces];
      face.index[num_faces] = d;
      face.face[num_faces] = t;
      c--;
    }
       
  calc_coeff ();
  calc_centroid ();
}


/*******************************************************************************
* Hexahedron::Hexahedron ()
*
* 	The default constructor, it sets up all values of the element to 0.
*******************************************************************************/

Hexahedron::Hexahedron ()
: my_index (0),
  my_centroid (0, 0, 0),
  num_faces (0),
  min (999999, 999999, 999999),
  max (0, 0, 0)
{
  int c;
    
  // Initialize the corners to 0.
  
  for (c = 8; c--;)
  {
    corner.node[c] = NULL;
    corner.index[c] = 0;
  }
  
  // Initialize the faces to 0.
  
  for (c = 6; c--;)
  {
    face.face[c] = NULL;
    face.index[c] = 0;
  }  
  
  calc_coeff ();
}


/*******************************************************************************
* void Hexahedron::calc_coeff ()
*
*
*     This function calculates the coefficients required for interpolation --
* these aide in finding s, t, and u.
*******************************************************************************/

void Hexahedron::calc_coeff ()
{
  int c, d;
  static double signs[] = {  1,  1,  1,  1,  1,  1,  1,  1,
                            -1,  1,  1, -1, -1,  1,  1, -1,
                            -1, -1,  1,  1, -1, -1,  1,  1,
                            -1, -1, -1, -1,  1,  1,  1,  1,
                             1, -1,  1, -1,  1, -1,  1, -1,
                             1, -1, -1,  1, -1,  1,  1, -1,
                             1,  1, -1, -1, -1, -1,  1,  1,
                            -1,  1, -1,  1,  1, -1,  1, -1};
  
  d = 0;
  
  for (v1 *= 0.0, c=0; c<8; c++) v1 += corner.node[c]->vector() * signs[d++];
  for (v2 *= 0.0, c=0; c<8; c++) v2 += corner.node[c]->vector() * signs[d++];
  for (v3 *= 0.0, c=0; c<8; c++) v3 += corner.node[c]->vector() * signs[d++];
  for (v4 *= 0.0, c=0; c<8; c++) v4 += corner.node[c]->vector() * signs[d++];
  for (v5 *= 0.0, c=0; c<8; c++) v5 += corner.node[c]->vector() * signs[d++];
  for (v6 *= 0.0, c=0; c<8; c++) v6 += corner.node[c]->vector() * signs[d++];
  for (v7 *= 0.0, c=0; c<8; c++) v7 += corner.node[c]->vector() * signs[d++];
  for (v8 *= 0.0, c=0; c<8; c++) v8 += corner.node[c]->vector() * signs[d++];

  v1 *= 0.125; v2 *= 0.125; v3 *= 0.125; v4 *= 0.125; 
  v5 *= 0.125; v6 *= 0.125; v7 *= 0.125; v8 *= 0.125; 
}


/*******************************************************************************
* void Hexahedron::calc_centroid ()
*
*
*     This function calculates the centroid of the volume.
*******************************************************************************/

void Hexahedron::calc_centroid ()
{
  Vector v(0,0,0);
  int i, j, k=0;
  double r;
  
  for (i = 0; i < 8; i++)
  {
    for (j = i; j--;)
      if (corner.index[i] == corner.index[j]) break;
      
    if (j < 0)
    {
      v += corner.node[i]->vector();
      k++;
    } 
  }
  
  v *= 1.0 / k;
  
  my_centroid = v.point();  
  
  my_radius = 0.0;
  
  for (i = 8; i--;)
  {
    r = (my_centroid- *corner.node[i]).length2();
    if (r > my_radius)
      my_radius = r;
      
    min=Min(min, *corner.node[i]);
    max=Max(max, *corner.node[i]);
  }
}


/*******************************************************************************
* void Hexahedron::find_stu (const Vector & p, double & s, double & t, double & u)
*
*
*     This function finds an s, t, and u coordinate for an x, y, and z 
* coordinate.
*******************************************************************************/

void Hexahedron::find_stu (const Vector & p, double & s, double & t, double & u)
{
  double ts, tt, tu, e, err = 1.0e+49;
  Vector s1, s2, s3, s4, t1, t2;
  Vector dc;
  double es, et, eu;
  
  dc = p - v1;
  if (Dot (dc, v2) > 0)
    es =  1.01;
  else
    es =  0.01;
  if (Dot (dc, v3) > 0)
    et =  1.01;
  else
    et =  0.01;
  if (Dot (dc, v4) > 0)
    eu =  1.01;
  else
    eu =  0.01;
  
  s = es - 0.51;
  t = et - 0.51;
  u = eu - 0.51;
  dc = p - v1 + v2*s + (v3 + v5*s)*t + (v4 + v6*s + (v7 + v8*s)*t)*u;
  ts = Dot (dc, v2 + v5*t + (v6 + v8*t)*u);
  tt = Dot (dc, v3 + v5*s + (v7 + v8*s)*u);
  tu = Dot (dc, v4 + v6*s + (v7 + v8*s)*t);
  if (ts < 0)
    es -= 0.5;
  if (tt < 0)
    et -= 0.5;
  if (tu < 0)
    eu -= 0.5;  
  
  for (ts = es - 0.51; ts < es; ts += 0.25)
  {
    s1 = v1 + v2*ts;
    s2 = v3 + v5*ts;
    s3 = v4 + v6*ts;
    s4 = v7 + v8*ts;
    for (tt = et - 0.51; tt < et; tt += 0.25)
    {
      t1 = s2*tt;
      t2 = s3 + s4*tt;
      for (tu = eu - 0.51; tu < eu; tu += 0.25)
      {
        e = (s1 + t1 + t2*tu - p).length2();
        if (e < err)
        {
          s = ts;
          t = tt;
          u = tu;
          err = e;
        }
      }
    }
  }

}


/*******************************************************************************
* ostream & operator << (ostream & o, const Hexahedron & h)
*
*	This function outputs the contents of the Hexahedron for debugging
* purposes only.
*******************************************************************************/

ostream & operator << (ostream & o, const Hexahedron & h)
{
  o << "Hexahedron " << (unsigned int) &h << ":" << " index=" << h.my_index
    << " num_faces=" << h.num_faces
    << endl;
    
  o << "    Nodes " << h.corner.index[0] << " " << h.corner.index[1] 
    << " " << h.corner.index[2] << " " << h.corner.index[3] 
    << " " << h.corner.index[4] << " " << h.corner.index[5] 
    << " " << h.corner.index[6] << " " << h.corner.index[7]
    << endl;
        
  o << "    Faces " << h.face.index[0] << " " << h.face.index[1] 
    << " " << h.face.index[2] << " " << h.face.index[3] 
    << " " << h.face.index[4] << " " << h.face.index[5]
    << endl;
    
  return o;
}


/*******************************************************************************
* void Pio (Piostream & p, Hexahedron & h)
*
*	Output or input the volume to the given io stream.  Note that if the
* data was input, an additional call must be made to finish setting up the
* pointers to the faces, etc.
*******************************************************************************/

void Pio (Piostream & p, Hexahedron & h)
{  
  int c;

  // Write out the data.

  p.begin_cheap_delim();  
  
  Pio(p, h.my_index);
  Pio(p, h.num_faces);
  
  for (c = 0; c < 8; c++)
    Pio (p, h.corner.index[c]);
    
  for (c = 0; c < 6; c++)
    Pio (p, h.face.index[c]);
      
  p.end_cheap_delim();
}


/*******************************************************************************
* void Pio (Piostream & p, Hexahedron * & h)
*
*	Output or input the volume to the given io stream.
*******************************************************************************/

void Pio (Piostream & p, Hexahedron * & h)
{
  // If we are reading, then we must assume the volume does not exist.
  //   Caution, this is prone to memory leaks.
  
  if (p.reading ())
    h = new Hexahedron ();
    
  // Use the other Pio routine.
  
  Pio (p, *h); 
}


/*******************************************************************************
* void Hexahedron::finish_read (HexMesh * m)
*
*     Set up pointers which were not stored in permanent storage.
*******************************************************************************/

void Hexahedron::finish_read (HexMesh * m)
{
  int c;
  
  for (c = 8; c--;)
    corner.node[c] = m->find_node (corner.index[c]);
    
  for (c = 6; c--;)
    face.face[c] = m->find_face (face.index[c]);
    
  calc_coeff (); 
  calc_centroid ();  
}


/*******************************************************************************
********************************************************************************
* HexMesh
********************************************************************************
*******************************************************************************/

/*******************************************************************************
* static Persistent* make_HexMesh()
*
* 	This function is provided so that the persistant base class can
* make objects of this type.
*******************************************************************************/

static Persistent* make_HexMesh()
{
    return scinew HexMesh ();
}


/*******************************************************************************
* HexMesh::HexMesh ()
*
* 	This default constructor sets up an empty mesh.  This default call is
* used prior to building a mesh from points and volumes or prior to using the
* pio calls to read in a mesh.
*******************************************************************************/

HexMesh::HexMesh ()
: highest_face_index (0),
  highest_node_index (0),
  classified (0)
{
}


/*******************************************************************************
* HexMesh::~HexMesh ()
*
* 	This destructor cleans up all memory used by this class.  First,
* all volumes are deleted, then faces, then nodes.
*******************************************************************************/

HexMesh::~HexMesh ()
{
  // Hashtable deletion takes care of this, this routine just a stub for now.
}


/*******************************************************************************
* HexMesh * HexMesh::clone ()
*
* 	This function creates a copy of this class instance and it returns
* a pointer to that instance.
*******************************************************************************/

HexMesh * HexMesh::clone ()
{
  // Not yet implemented.
  
  return NULL;
}


/*******************************************************************************
* int HexMesh::add_node (int index, double x, double y, double z)
*
* 	Add a point to the point hash table.  No two points may share the
* same index -- add_node will not allow this.  Returns 0 upon success, -1
* upon failure.
*******************************************************************************/

int HexMesh::add_node (int index, double x, double y, double z)
{
  HexNode * n;
  
  // Look up given index to make sure it does not exist.
  
  if (find_node (index) != NULL)
    return -1;
  
  // Make a new node, add it to the hash table, and return.
  
  n = new HexNode (index, x, y, z);
  
  node_set.insert (index, n);
  
  if (index > highest_node_index)
    highest_node_index = index;
  
  return index;
}


/*******************************************************************************
* HexNode * HexMesh::find_node (int index)
*
* 	If a node exists in the mesh with the given index, that node is
* returned.  Otherwise, NULL is returned.
*******************************************************************************/

HexNode * HexMesh::find_node (int index)
{
  HexNode * found = NULL;

  // Look up the index.  Return NULL if a node does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  if (node_set.lookup (index, found) == 0)
    return NULL;
    
  return found;
}


/*******************************************************************************
* int HexMesh::add_face (int index, int e, FourHexNodes & f)
*
* 	Add a face to the mesh.  No two faces may share the
* same index -- add_face will not allow this.  Returns 0 upon success, -1
* upon failure.  (e = index of element which is adding this face.)
*******************************************************************************/

int HexMesh::add_face (int index, int e, FourHexNodes & f)
{
  HexFace * h;
  int c;
  
  // Look up given index to make sure it does not exist.
  
  if (find_face (index) != NULL)
    return -1;

  // Make sure the nodes exist.
  
  for (c = 4; c--;)
  {
    f.node[c] = find_node (f.index[c]);
    if (f.node[c] == NULL)
      return -1;
  }
      
  // Bump the highest index tracker.
  
  if (highest_face_index < index)
    highest_face_index = index;

  // Make a new face, add it to the hash table, and return.
  
  h = new HexFace (index, e, f, this);
  
  face_set.insert (index, h);
  neighbor_set.insert (f, h);
  
  return index;
}


/*******************************************************************************
* int HexMesh::add_face (int e, FourHexNodes & f)
*
* 	Add a face to the mesh.  Since no index was specified, the new face
* will be given an index one larger than the largest one added so far.
*******************************************************************************/

int HexMesh::add_face (int e, FourHexNodes & f)
{
  return add_face (highest_face_index + 1, e, f);
}


/*******************************************************************************
* HexFace * HexMesh::find_face (int index)
*
* 	If a face exists in the mesh with the given index, that face is
* returned.  Otherwise, NULL is returned.
*******************************************************************************/

HexFace * HexMesh::find_face (int index)
{
  HexFace * found = NULL;

  // Look up the index.  Return NULL if the face does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  if (face_set.lookup (index, found) == 0)
    return NULL;
    
  return found;
}


/*******************************************************************************
* HexFace * HexMesh::find_face (FourHexNodes & f)
*
* 	This function looks for a face using the nodes as the key.  If it is 
* found, it is returned.  Otherwise, NULL is returned.
*******************************************************************************/

HexFace * HexMesh::find_face (FourHexNodes & f)
{
  HexFace * found = NULL;

  // Look up the index.  Return NULL if the face does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  if (neighbor_set.lookup (f, found) == 0)
    return NULL;
    
  return found;
}


/*******************************************************************************
* int HexMesh::add_element (int index,  EightHexNodes & e)
*
* 	Add an element to the mesh.  No two volume elements may share the
* same index -- add_element will not allow this.  Returns 0 upon success, -1
* upon failure.
*******************************************************************************/

int HexMesh::add_element (int index, EightHexNodes & e)
{
  Hexahedron * h;
  int c;
  
  // Look up given index to make sure it does not exist.
  
  if (find_element (index) != NULL)
    return -1;

  // Make sure the nodes exist.
  
  for (c = 8; c--;)
  {
    e.node[c] = find_node (e.index[c]);
    if (e.node[c] == NULL)
      return -1;
  }
      
  // Make a new volume element, add it to the hash table, and return.
  
  h = new Hexahedron (index, this, e);
  
  element_set.insert (index, h);
  
  return index;
}


/*******************************************************************************
* Hexahedron * HexMesh::find_element (int index)
*
* 	If an element exists in the mesh with the given index, that element is
* returned.  Otherwise, NULL is returned.
*******************************************************************************/

Hexahedron * HexMesh::find_element (int index)
{
  Hexahedron * found = NULL;

  // Look up the index.  Return NULL if the element does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  if (element_set.lookup (index, found) == 0)
    return NULL;
    
  return found;
}


/*******************************************************************************
* Classify operator
*
*     This function builds a KD tree of elements.
*******************************************************************************/

void PushDown (KDTree * c, Hexahedron * h, int level)
{
  Point mmin, mmax;
  double dx, dy, dz;
  int side;
  
  switch (c->split)
  {
    case 0:
      dx = (c->min.x() + c->max.x()) / 2;
      if (h->min.x() <= dx && h->max.x() >= dx)
        side = 0;
      else if (h->min.x() <= dx && h->max.x() <= dx)
      {
        mmin = c->min;
        mmax = c->max;
        mmax.x (dx);
        side = 1;
      }
      else
      {
        mmin = c->min;
        mmax = c->max;
        mmin.x (dx);
        side = 2;
      }
      break;
    case 1:
      dy = (c->min.y() + c->max.y()) / 2;
      if (h->min.y() <= dy && h->max.y() >= dy)
        side = 0;
      else if (h->min.y() <= dy && h->max.y() <= dy)
      {
        mmin = c->min;
        mmax = c->max;
        mmax.y (dy);
        side = 1;
      }
      else
      {
        mmin = c->min;
        mmax = c->max;
        mmin.y (dy);
        side = 2;
      }
      break;
    case 2:
      dz = (c->min.z() + c->max.z()) / 2;
      if (h->min.z() <= dz && h->max.z() >= dz)
        side = 0;
      else if (h->min.z() <= dz && h->max.z() <= dz)
      {
        mmin = c->min;
        mmax = c->max;
        mmax.z (dz);
        side = 1;
      }
      else
      {
        mmin = c->min;
        mmax = c->max;
        mmin.z (dz);
        side = 2;
      }
      break;
  }
      
  if (level >= 20)
    side = 0;    
      
  switch (side)
  {
    case 0:
      if (level >= 20)
        c->here.add (h);
      else
      {
        if (c->low == NULL)
        {
          c->low = new KDTree;
          c->low->min = mmin;
          c->low->max = mmax;
          c->low->split = (c->split+1) % 3;
          c->low->low = c->low->high = NULL;
        }
        PushDown (c->low, h, level+1);
        if (c->high == NULL)
        {
          c->high = new KDTree;
          c->high->min = mmin;
          c->high->max = mmax;
          c->high->split = (c->split+1) % 3;
          c->high->low = c->high->high = NULL;
        }
        PushDown (c->high, h, level+1);
      }
      break;
    case 1:
      if (c->low == NULL)
      {
        c->low = new KDTree;
        c->low->min = mmin;
        c->low->max = mmax;
        c->low->split = (c->split+1) % 3;
        c->low->low = c->low->high = NULL;
      }
      PushDown (c->low, h, level+1);
      break;
    case 2:
      if (c->high == NULL)
      {
        c->high = new KDTree;
        c->high->min = mmin;
        c->high->max = mmax;
        c->high->split = (c->split+1) % 3;
        c->high->low = c->high->high = NULL;
      }
      PushDown (c->high, h, level+1);
      break;
  }    
}

void HexMesh::classify ()
{
  Hexahedron * h;
  HashTable<int, Hexahedron *> * hxhtp = & element_set;
  HashTableIter<int, Hexahedron *> hx (hxhtp);
  
  cout << "Building KD tree.\n";
  
  classified = 1;
  
  KD.min.x(999999);
  KD.min.y(999999);
  KD.min.z(999999);
  KD.max.x(0);
  KD.max.y(0);
  KD.max.z(0);
  
  for (hx.first (); hx.ok (); ++hx)
  {
    h = hx.get_data();
    KD.min = Min (KD.min, h->min);
    KD.max = Max (KD.max, h->max);
  }
  
  KD.split = 0;
  KD.low = KD.high = NULL;
  
  for (hx.first (); hx.ok (); ++hx)
  {
    h = hx.get_data();
    
    PushDown (&KD, h, 1);    
  }
  
  cout << "KD tree done.\n";
}

/*******************************************************************************
* Locate operator
*
*     This function returns the index of the Hexahedron that contains the
* given point, if any.  The node to start searching at is specified.
*******************************************************************************/

int HexMesh::locate (const Point& P, int & idx)
{
  Hexahedron * h;
  int c, max_iter, m, next, fail;
  double smallest, dist;
      
  if (idx < 1)
    idx = 1;

  // Do smart search here, artificially limit the depth.

  for (max_iter = 150; idx >= 0 && max_iter--;)
  {
    h = find_element (idx);
    
    if (h == NULL)
      break;
    
    fail = next = 0;
    smallest = -1e-10;
    
    for (c = h->surface_count (); c--;)
    {
      dist = h->surface(c)->dist(P);

      if (dist < smallest)
      {
        fail ++;
        m = h->surface(c)->neighbor_index ();
        
        if (m != 0)
        {
          smallest = dist;
          next = m;
        }
      }
    }

    if (!fail)
    {
      return idx;
    }
    else  
      idx = next;
  }
  
  // Smart search failed -- do stupid search.

  // Stupid search only needed for volumes with cavaties or concave components.
  // Skip this step now.

  if (!classified)
    classify ();

  KDTree * k;
  
  k = &KD;
  
  if (P.x() < KD.min.x() || P.y() < KD.min.y() || P.z() < KD.min.z() || 
      P.x() > KD.max.x() || P.y() > KD.max.y() || P.z() > KD.max.z())
    return idx = -1;
    
  while (k != NULL)
  {
    for (m = k->here.size (); --m >= 0;)
    {
      h = k->here[m];
      if (
          (h->centroid().x() - P.x()) * (h->centroid().x() - P.x()) + 
          (h->centroid().y() - P.y()) * (h->centroid().y() - P.y()) + 
          (h->centroid().z() - P.z()) * (h->centroid().z() - P.z()) - h->radius () > 0)
        continue; 
      for (c = h->surface_count (); c--;)
        if (h->surface(c)->dist(P) < -1e-10)
          break;
      if (c == -1)
        return (idx = h->index());   
    }
    
    switch (k->split)
    {
      case 0:
        if (P.x() < (k->min.x() + k->max.x()) / 2)
          k = k->low;
        else 
          k = k->high;
          break;
      case 1:
        if (P.y() < (k->min.y() + k->max.y()) / 2)
          k = k->low;
        else 
          k = k->high;
          break;
      case 2:
        if (P.z() < (k->min.z() + k->max.z()) / 2)
          k = k->low;
        else 
          k = k->high;
          break;
    }
  }

  return idx = -1;
#endif
}


/*******************************************************************************
* Interpolation operator
*
*     This function returns the value at the given data point.  
*******************************************************************************/

double HexMesh::interpolate (const Point & p, const Array1<double> & data, int & start)
{
  Hexahedron * h;
  double s, t, u, sm1, sp1, tm1, tp1, um1, up1;

  // Find which node has this point.
  
  locate (p, start);
  if (start < 0)
    return -1;
    
  h = find_element (start);

  if (h == NULL)
    return -1;
    

  // Patch to deal with tetra.
  
  /*
  if (h->node_index(0) == h->node_index(1) &&
      h->node_index(1) == h->node_index(2) &&
      h->node_index(2) == h->node_index(3))
  {
    cout << "Tetra.\n";
    Point p1(h->node(0));
    Point p2(h->node(4));
    Point p3(h->node(6));
    Point p4(h->node(7));
    double x1=p1.x();
    double y1=p1.y();
    double z1=p1.z();
    double x2=p2.x();
    double y2=p2.y();
    double z2=p2.z();
    double x3=p3.x();
    double y3=p3.y();
    double z3=p3.z();
    double x4=p4.x();
    double y4=p4.y();
    double z4=p4.z();
    double a1=+x2*(y3*z4-y4*z3)+x3*(y4*z2-y2*z4)+x4*(y2*z3-y3*z2);
    double a2=-x3*(y4*z1-y1*z4)-x4*(y1*z3-y3*z1)-x1*(y3*z4-y4*z3);
    double a3=+x4*(y1*z2-y2*z1)+x1*(y2*z4-y4*z2)+x2*(y4*z1-y1*z4);
    double a4=-x1*(y2*z3-y3*z2)-x2*(y3*z1-y1*z3)-x3*(y1*z2-y2*z1);
    double iV6=1./(a1+a2+a3+a4);

    double b1=-(y3*z4-y4*z3)-(y4*z2-y2*z4)-(y2*z3-y3*z2);
    double c1=+(x3*z4-x4*z3)+(x4*z2-x2*z4)+(x2*z3-x3*z2);
    double d1=-(x3*y4-x4*y3)-(x4*y2-x2*y4)-(x2*y3-x3*y2);
    double value = data[h->node_index(0)] * iV6*(a1+b1*p.x()+c1*p.y()+d1*p.z());
    double b2=+(y4*z1-y1*z4)+(y1*z3-y3*z1)+(y3*z4-y4*z3);
    double c2=-(x4*z1-x1*z4)-(x1*z3-x3*z1)-(x3*z4-x4*z3);
    double d2=+(x4*y1-x1*y4)+(x1*y3-x3*y1)+(x3*y4-x4*y3);
    value += data[h->node_index(4)] * iV6*(a2+b2*p.x()+c2*p.y()+d2*p.z());
    double b3=-(y1*z2-y2*z1)-(y2*z4-y4*z2)-(y4*z1-y1*z4);
    double c3=+(x1*z2-x2*z1)+(x2*z4-x4*z2)+(x4*z1-x1*z4);
    double d3=-(x1*y2-x2*y1)-(x2*y4-x4*y2)-(x4*y1-x1*y4);
    value += data[h->node_index(6)] * iV6*(a3+b3*p.x()+c3*p.y()+d3*p.z());
    double b4=+(y2*z3-y3*z2)+(y3*z1-y1*z3)+(y1*z2-y2*z1);
    double c4=-(x2*z3-x3*z2)-(x3*z1-x1*z3)-(x1*z2-x2*z1);
    double d4=+(x2*y3-x3*y2)+(x3*y1-x1*y3)+(x1*y2-x2*y1);
    return value + data[h->node_index(7)] * iV6*(a4+b4*p.x()+c4*p.y()+d4*p.z());
  }
*/

  // Find interpolants.
  
  h->find_stu (p.vector(), s, t, u);
  sm1 = s - 1;
  sp1 = s + 1;
  tm1 = t - 1;
  tp1 = t + 1;
  um1 = u - 1;
  up1 = u + 1;
      
  // Return an interpolated value.
  
  return ( - sm1*tm1*um1*data[h->node_index(0)]
           + sp1*tm1*um1*data[h->node_index(1)]
           - sp1*tp1*um1*data[h->node_index(2)]
           + sm1*tp1*um1*data[h->node_index(3)]
           + sm1*tm1*up1*data[h->node_index(4)]
           - sp1*tm1*up1*data[h->node_index(5)]
           + sp1*tp1*up1*data[h->node_index(6)]
           - sm1*tp1*up1*data[h->node_index(7)]) * 0.125;
}


/*******************************************************************************
* Bounding box operator
*
*     This function finds the bounding box of the volume given by this mesh.
*******************************************************************************/

void HexMesh::get_bounds (Point& min, Point& max)
{
  HashTable<int, HexNode *> * hnhtp = & node_set;
  HashTableIter<int, HexNode *> hn (hnhtp);

  // Loop through the nodes looking for min/max.  Assumes at least one node exists.

  hn.first ();
  min = max = *hn.get_data ();

  for (++hn; hn.ok (); ++hn)
  {
    min=Min(min, *hn.get_data ());
    max=Max(max, *hn.get_data ());
  }
}    


/*******************************************************************************
* ostream & operator << (ostream & o, const HexMesh & m)
*
*	This function outputs the contents of the HexMesh for debugging
* purposes only.
*******************************************************************************/

ostream & operator << (ostream & o, HexMesh & m)
{
  HashTable<int, HexNode *> * hnhtp = & m.node_set;
  HashTableIter<int, HexNode *> hn (hnhtp);

  HashTable<int, HexFace *> * hfhtp = & m.face_set;
  HashTableIter<int, HexFace *> hf (hfhtp);

  HashTable<int, Hexahedron *> * hxhtp = & m.element_set;
  HashTableIter<int, Hexahedron *> hx (hxhtp);

  // Print a header.

  o << "HexMesh " << (unsigned int) &m << endl;
    
  // Print the nodes.
  
  o << "  Nodes (" << m.node_set.size() << "):" << endl;
  
  for (hn.first (); hn.ok (); ++hn)
    o << "  " << * hn.get_data();
    
  // Print the faces.
  
  o << "  Faces (" << m.face_set.size() << "):" << endl;
  
  for (hf.first (); hf.ok (); ++hf)
    o << "  " << * hf.get_data();
    
  // Print the volumes.
  
  o << "  Hexahedrons (" << m.element_set.size() << "):" << endl;
  
  for (hx.first (); hx.ok (); ++hx)
    o << "  " << * hx.get_data();
    
  // Done dumping the mesh.  
    
  return o;
}


/*******************************************************************************
* void HexMesh::io (Piostream & p)
*
*	Output or input this mesh to the given io stream.  (All sub-objects
* are subsequently output.)
*******************************************************************************/

void HexMesh::io (Piostream & p)
{
  HashTable<int, HexFace *> * hfhtp = & face_set;
  HashTableIter<int, HexFace *> hf (hfhtp);
  HashTable<int, Hexahedron *> * hxhtp = & element_set;
  HashTableIter<int, Hexahedron *> hx (hxhtp);
  int version;
  
  // Set up a header and get/put our version.
  
  version = p.begin_class("HexMesh", HEXMESH_VERSION);
  
  if (version == 1)
  {
    Pio (p, highest_face_index);
    Pio (p, highest_node_index);
  
    // Deal with the mesh data.
  
    Pio (p, node_set);
    Pio (p, face_set);
    Pio (p, element_set);
  
    // End the data io with a post-amble.
  
    p.end_class();

    // If we just read the mesh, some additional processing may be required.
  
    if (p.reading ())
    {
      for (hx.first (); hx.ok (); ++hx)
        hx.get_data()->finish_read (this);
      for (hf.first (); hf.ok (); ++hf)
      {
        hf.get_data()->finish_read (this);
        neighbor_set.insert ( hf.get_data()->corner_set(), hf.get_data());
      }
    }
  }
}

void HexMesh::get_boundary_lines(Array1<Point>& lines)
{
    NOT_FINISHED("HexMesh::get_boundary_lines");
}

