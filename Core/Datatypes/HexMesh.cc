/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Datatypes/HexMesh.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>

#include <Core/Persistent/PersistentSTL.h>

//#include <iostream>
//#include <fstream>

using std::cout;
using std::endl;
using std::ostream;

using std::map;

/*******************************************************************************
********************************************************************************
* Global variables and forward declarations.
********************************************************************************
*******************************************************************************/

namespace SCIRun {

static Persistent* make_HexMesh();

PersistentTypeID
HexMesh::type_id("HexMesh", "Datatype", make_HexMesh);


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

std::ostream & operator << (std::ostream & o, const HexNode & n)
{
  o << "HexNode " << (void *) &n << ": x=" << n.x() << " y=" << n.y() 
    << " z=" << n.z() << " index=" << n.my_index
    << endl;
    
  return o;
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
  // Make a copy of the corners.
  
  corner = f;

  // Recalculate the normal, centroid, and planar adherence.
  
  calc_face (m);  
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

void HexFace::calc_face (HexMesh * m)
{


  double d;

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
  
  planar_adherence  = Abs (Dot (my_normal, my_centroid - *corner.node[0]));
  planar_adherence += Abs (Dot (my_normal, my_centroid - *corner.node[1]));
  planar_adherence += Abs (Dot (my_normal, my_centroid - *corner.node[2]));
  planar_adherence += Abs (Dot (my_normal, my_centroid - *corner.node[3]));
  
  planar_adherence /= 4.0;
  
  // Set d.
  
  my_d = - Dot (my_normal, my_centroid);
  
  // Reorient the normal.
  
  d = Dot (my_normal, m->find_element (my_contains_index)->centroid()) + my_d;
  if (d < 0)
  {
    my_normal *= -1.0; 
    my_d *= -1.0;
  } 
        
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

std::ostream & operator << (std::ostream & o, const HexFace & f)
{
  o << "HexFace " << (void *) &f << ": index=" << f.my_index
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
 * void HexFace::finish_read (HexMesh * m)
 *
 *     Set up pointers which were not stored in permanent storage.
 *******************************************************************************/

void HexFace::finish_read (HexMesh * m)
{
  int c;
  
  for (c = 8; c--;)
    corner.node[c] = m->find_node (corner.index[c]);

  calc_face (m);
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
  : my_index (index), corner(e),
    num_faces (6),
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

std::ostream & operator << (std::ostream & o, const Hexahedron & h)
{
  o << "Hexahedron " << (void *) &h << ":" << " index=" << h.my_index
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
  : classified (0),
    highest_face_index (0),
    highest_node_index (0)
{
}


HexMesh::HexMesh(const HexMesh &copy) :
  node_set(copy.node_set),
  element_set(copy.element_set),
  face_set(copy.face_set),
  neighbor_set(copy.neighbor_set),
  classified(copy.classified),
  KD(copy.KD),
  highest_face_index(copy.highest_face_index),
  highest_node_index(copy.highest_node_index),
  highest_element_index(copy.highest_element_index)
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

HexMesh *HexMesh::clone()
{
  return scinew HexMesh(*this);
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
  
  node_set[index] = n;
  
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
  MapIntHexNode::iterator found;

  // Look up the index.  Return NULL if a node does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  found = node_set.find(index);
  if (found == node_set.end()) return NULL;
  return (*found).second;
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
  
  face_set[index] = h;
  neighbor_set[f] = h;
  
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
  MapIntHexFace::iterator found;

  // Look up the index.  Return NULL if the face does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  found = face_set.find(index);
  if (found == face_set.end()) return NULL;
    
  return (*found).second;
}


/*******************************************************************************
 * HexFace * HexMesh::find_face (FourHexNodes & f)
 *
 * 	This function looks for a face using the nodes as the key.  If it is 
 * found, it is returned.  Otherwise, NULL is returned.
 *******************************************************************************/

HexFace * HexMesh::find_face (FourHexNodes & f)
{
  MapFourHexNodesHexFace::iterator found;

  // Look up the index.  Return NULL if the face does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  found = neighbor_set.find(f);
  if (found == neighbor_set.end()) return NULL;
    
  return (*found).second;
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
  
  element_set[index] = h;
  
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
  MapIntHexahedron::iterator found;

  // Look up the index.  Return NULL if the element does not exist.  Otherwise,
  //   'found' gets set up -- return it.
  
  found = element_set.find(index);
  if (found == element_set.end()) return NULL;
    
  return (*found).second;
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
  int side=0;
  
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
  MapIntHexahedron::iterator hx;
  
  cout << "Building KD tree.\n";
  
  classified = 1;
  
  KD.min.x(999999);
  KD.min.y(999999);
  KD.min.z(999999);
  KD.max.x(0);
  KD.max.y(0);
  KD.max.z(0);
  
  for (hx = element_set.begin(); hx != element_set.end(); hx++) {
    h = (*hx).second;
    KD.min = Min (KD.min, h->min);
    KD.max = Max (KD.max, h->max);
  }
  
  KD.split = 0;
  KD.low = KD.high = NULL;
  
  for (hx = element_set.begin(); hx != element_set.end(); hx++) {
    h = (*hx).second;
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

bool HexMesh::locate(int *idx, const Point& P)
{
  Hexahedron * h;
  int c, max_iter, m, next, fail;
  double smallest, dist;
      
  if (*idx < 1)
    *idx = 1;

  // Do smart search here, artificially limit the depth.

  for (max_iter = 150; *idx >= 0 && max_iter--;)
  {
    h = find_element (*idx);
    
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
      return *idx;
    }
    else  
      *idx = next;
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
  {
    *idx = -1;
    return false;
  }
    
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
      {
	*idx = h->index();
	return true;
      }
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
  *idx = -1;
  return false;
}


/*******************************************************************************
 * Interpolation operators
 *
 *     This function returns the value at the given data point.  
 *******************************************************************************/

double HexMesh::interpolate (const Point & p, const Array1<double> & data, int & start)
{
  Hexahedron * h;
  double s, t, u, sm1, sp1, tm1, tp1, um1, up1;

  // Find which node has this point.
  
  locate (&start, p);
  if (start < 0)
    return -1;
    
  h = find_element (start);

  if (h == NULL)
    return -1;
    
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

double HexMesh::interpolate (const Point & P, const Array1<Vector> & data,
			     Vector &v, int & start)
{
  Hexahedron * h;
  double s, t, u, sm1, sp1, tm1, tp1, um1, up1;

  // Find which node has this point.

  locate (&start, P);
  if (start < 0)
    return -1;

  h = find_element (start);

  if (h == NULL)
    return -1;


  // Find interpolants.

  h->find_stu (P.vector(), s, t, u);
  sm1 = s - 1;
  sp1 = s + 1;
  tm1 = t - 1;
  tp1 = t + 1;
  um1 = u - 1;
  up1 = u + 1;

  // Assign the interpolated vector

  v = (- data[h->node_index(0)]*(sm1*tm1*um1)
       + data[h->node_index(1)]*(sp1*tm1*um1)
       - data[h->node_index(2)]*(sp1*tp1*um1)
       + data[h->node_index(3)]*(sm1*tp1*um1)
       + data[h->node_index(4)]*(sm1*tm1*up1)
       - data[h->node_index(5)]*(sp1*tm1*up1)
       + data[h->node_index(6)]*(sp1*tp1*up1)
       - data[h->node_index(7)]*(sm1*tp1*up1)) * 0.125;

  // Return something that i don't understand

  return 0.0;
}

/*******************************************************************************
 * Bounding box operator
 *
 *     This function finds the bounding box of the volume given by this mesh.
 *******************************************************************************/

void HexMesh::get_bounds (Point& min, Point& max)
{
  MapIntHexNode::iterator hn;

  // Loop through the nodes looking for min/max.  Assumes at least one node exists.
  hn = node_set.begin();
  min = max = *(*hn).second;

  for (++hn; hn != node_set.end(); ++hn) {
    min=Min( min, *(*hn).second );
    max=Max( max, *(*hn).second );
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
  HexMesh::MapIntHexNode::iterator hn;
  HexMesh::MapIntHexFace::iterator hf;
  HexMesh::MapIntHexahedron::iterator hx;

  // Print a header.

  o << "HexMesh " << (void *) &m << endl;
    
  // Print the nodes.
  
  o << "  Nodes (" << m.node_set.size() << "):" << endl;
  
  for (hn = m.node_set.begin(); hn != m.node_set.end(); ++hn) {
    o << "  " << *((*hn).second);
  }
    
  // Print the faces.
  
  o << "  Faces (" << m.face_set.size() << "):" << endl;
  
  for (hf = m.face_set.begin(); hf != m.face_set.end(); ++hf) {
    o << "  " << *((*hf).second);
  }
    
  // Print the volumes.
  
  o << "  Hexahedrons (" << m.element_set.size() << "):" << endl;
  
  for (hx = m.element_set.begin(); hx != m.element_set.end(); ++hx) {
    o << "  " << *((*hx).second);
  }
    
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
      finish ();
    }
  }
}

//----------------------------------------------------------------------
void HexMesh::finish ()
{
  MapIntHexFace::iterator hf;
  MapIntHexahedron::iterator hx;
  
				// If we just read the mesh, some
				// additional processing may be
				// required.
  
  for (hx = element_set.begin(); hx != element_set.end(); ++hx) {
    (*hx).second->finish_read(this);
  }
  
  for (hf = face_set.begin(); hf != face_set.end(); ++hf) {
    (*hf).second->finish_read(this);
    neighbor_set[(*hf).second->corner_set()] = (*hf).second;
  }
  
  if (!classified) classify();
}

//----------------------------------------------------------------------
void HexMesh::get_boundary_lines(Array1<Point>&)
{
  NOT_FINISHED("HexMesh::get_boundary_lines");
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

} // End namespace SCIRun
