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
 *  Sourced from:
 *    Mesh.h
 *
 *  Copyright (C) 1997 SCI Group
 */

/******************************************************************************
* Version control
******************************************************************************/

#ifndef SCI_project_HexMesh_h
#define SCI_project_HexMesh_h 1

/******************************************************************************
* Includes
******************************************************************************/

#include <Core/share/share.h>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

#include <map.h>

/******************************************************************************
* Class & function forward declarations and type declarations
******************************************************************************/
namespace SCIRun {


class HexNode;
class HexFace;
class Hexahedron;
class HexMesh;

/******************************************************************************
 * Handles
 ******************************************************************************/

typedef LockingHandle<HexMesh> HexMeshHandle;


/******************************************************************************
 * Index/pointer array structures  (Used to make copying/grouping easy.)
 ******************************************************************************/

struct FourHexNodes
{
  int index[4];
  HexNode * node[4];
  
  int operator == (const FourHexNodes & f)
  {  int used[4] = { 0, 0, 0, 0 }, c, d, unmatched = 4;
  for (c = 4; c--;)
    for (d = 4; d--;)
      if (!used[d] && f.index[c] == index[d])
      {  unmatched--;  used[d]++;  break; }      
  return unmatched == 0;
  };

  /*
    int hash (int h) const
    { return (((index[0] + index[1] + index[2] + index[3]) + 
    (index[0] ^ index[1] ^ index[2] ^ index[3]) +
    (index[0] * index[1] * index[2] * index[3])) &
    0x7fffffff) % h; };
  */

				// less than operator, required for
				// use with STL maps - replaces hash()
  bool operator<(const FourHexNodes& f) const {
    return ((index[0]+index[1]+index[2]+index[3]) <
	    (f.index[0]+f.index[1]+f.index[2]+f.index[3]));
  }
  
};


struct EightHexNodes
{
  int index[8];
  HexNode * node[8];
};


struct SixHexFaces
{
  int index[6];
  HexFace * face[6];
};


struct KDTree
{
  Point min, max;
  KDTree *low, *high;
  int split;             // 0 = x, 1 = y, 2 = z;
  
  Array1 <Hexahedron *> here;
};


/*****************************************************************************
 * Class actual declarations
 *****************************************************************************/
// This class contains one point, similar to Node in mesh.h.

class SCICORESHARE HexNode : public Point
{
private:
  
  int my_index;		// The index of this node (in the hash table).
    
public:

  HexNode (int, double x, double y, double z);
  HexNode ();
      
  inline void set_index (int i) { my_index = i; };
  inline int index () { return my_index; };
    
  friend SCICORESHARE std::ostream & operator << (std::ostream & o, const HexNode & n);
  friend SCICORESHARE void Pio( Piostream & p, HexNode & );
  friend SCICORESHARE void Pio( Piostream & p, HexNode * & );
};



//----------------------------------------------------------------------
// This class defines one surface, similar to face in mesh.h.
//   Note that many fields are INVALID for degenerate faces (lines, points).
class SCICORESHARE HexFace
{
private:
      
  int my_index;		// The index of this node (in the hash table).
  int my_contains_index;      // Index of the hexahedron that contains this face.
  int my_neighbor_index;      // Index of other hexahedron that has this face.

  Point my_centroid;		// A point (ANY point) on the face of the plane.
  Vector my_normal;		// A normal to the plane.
  double my_d;		// Ax+By+Cz+d = 0;
  double planar_adherence;    // An average distance of corners from the plane.

  FourHexNodes corner;	// Index/pointer array to corner nodes.

  void calc_face (HexMesh *);		// Calculates the centroid and the normal.

public:

  HexFace (int, int, FourHexNodes & f, HexMesh * m);
  HexFace ();

  // Setup and query functions.
    
  inline int get_corner (int i) { return corner.index[i%4]; }
  int is_corner (int i);	     // Returns true if the index matches a corner.
  int compare (FourHexNodes & f);  // Returns true if the corners match.
  FourHexNodes corner_set () { return corner; };  // For hashing.
  
  inline void set_index (int i) { my_index = i; };
  inline int index () { return my_index; };
  inline void set_contains_index (int i) { my_contains_index = i; };
  inline int contains_index () { return my_contains_index; };
  inline void set_neighbor_index (int i) { my_neighbor_index = i; };
  inline int neighbor_index () { return my_neighbor_index; };
    
  // Property access functions.
    
  inline Point centroid () { return my_centroid; };
  inline Vector normal () { return my_normal; };
  inline double planar_fit () { return planar_adherence; };
  inline double d () { return my_d; };
    
  double dist (const Point & P);  // Used for finding if points lie in a volume.
    
  // I/o functions.
    
  friend SCICORESHARE std::ostream & operator << (std::ostream & o, const HexFace & n);
  friend SCICORESHARE void Pio( Piostream & p, HexFace & );
  friend SCICORESHARE void Pio( Piostream & p, HexFace * & );
  void finish_read (HexMesh * m);
};

//----------------------------------------------------------------------
// This class defines one volume unit, similar to element in mesh.h.
class SCICORESHARE Hexahedron
{  
private:
  
  int my_index;		// The index of this element.
    
  EightHexNodes corner;       // The nodes for this volume.
    
  Point my_centroid;          // The centroid of this volume.
  double my_radius;            // A minimum radius for this volume.
    
  SixHexFaces face;		// Pointer array to faces.
  int num_faces;              // Actual number of non-trivial faces.
    
  Vector v1, v2, v3, v4, v5, v6, v7, v8;  // Used in interpolation.
          
  void calc_coeff ();         // Used in interpolation.
  void calc_centroid ();      // Used in interpolation.
                
public:

  Point min, max;

  Hexahedron (int, HexMesh * m, EightHexNodes & e);
  Hexahedron ();
    
  inline void set_index (int i) { my_index = i; };
  inline int index () { return my_index; };
        
  inline int surface_count () { return num_faces; };
  inline int surface_index (int i) { return face.index[i%6]; };
  inline HexFace * surface (int i) { return face.face[i%6]; };
  inline int node_index (int i) { return corner.index[i%8]; };
  inline const Point & node (int i) { return *corner.node[i%8]; };
  inline const Point & centroid () { return my_centroid; };
  inline double radius () { return my_radius; };
        
  void find_stu (const Vector & P, double & s, double & t, double & u);    
        
  friend SCICORESHARE std::ostream & operator << (std::ostream & o, const Hexahedron & n);
  friend SCICORESHARE void Pio( Piostream & p, Hexahedron & );
  friend SCICORESHARE void Pio( Piostream & p, Hexahedron * & );
  void finish_read (HexMesh * m);
};


class SCICORESHARE HexMesh : public Datatype
{
  //public:
protected:

  typedef map<int, HexNode*, less<int> >	MapIntHexNode;
  typedef map<int, Hexahedron*, less<int> >	MapIntHexahedron;
  typedef map<int, HexFace*, less<int> >	MapIntHexFace;
  typedef map<FourHexNodes, HexFace*, less<FourHexNodes> >
  MapFourHexNodesHexFace;
  
  MapIntHexNode			node_set;
  MapIntHexahedron		element_set;
  MapIntHexFace			face_set;
  MapFourHexNodesHexFace	neighbor_set;
  
  int classified;
  KDTree KD;
  
  int highest_face_index;
  int highest_node_index;
  int highest_element_index;
  
public:
  
  // Constructors and destructors.
  HexMesh ();
  HexMesh(const HexMesh &copy);
  virtual HexMesh *clone();
  virtual ~HexMesh ();
  
  // Setup functions
  int       add_node    (int index, double x, double y, double z);
  HexNode * find_node   (int index);
  int       high_node   () { return highest_node_index; };
  
  int       add_face    (int index, int e, FourHexNodes & f); 
  int       add_face    (int e, FourHexNodes & f); 
  HexFace * find_face   (int index);
  HexFace * find_face   (FourHexNodes & f);
  int       high_face   () { return highest_face_index; };
  
  int          add_element    (int index, EightHexNodes & e); 
  Hexahedron * find_element   (int index);

private:  
  // Access functions
  void classify();
  void finish();

public:
  
  bool locate(int *start, const Point &p);
  
  double interpolate (const Point & P,
		      const Array1<double> & data, int & start);
  
  double interpolate (const Point & P,
		      const Array1<Vector> & data, Vector & v, int & start);
  
  void get_bounds( Point& min, Point& max );
  void get_boundary_lines( Array1<Point>& lines );
  
  // Io functions
  friend SCICORESHARE std::ostream& operator<< (std::ostream& o, HexMesh& m);
  virtual void io (Piostream& p);
  static PersistentTypeID type_id;

};

} // End namespace SCIRun

#endif
