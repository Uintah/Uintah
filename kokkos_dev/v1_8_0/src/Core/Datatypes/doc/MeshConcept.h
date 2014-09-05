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
 *  MeshConcept.h: The Mesh Concept, documentation
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef SCI_project_MeshConcept_h
#define SCI_project_MeshConcept_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;
using std::vector;

//! The Mesh Concept, a sample mesh with all the required interface.
/*! This is all the things that belong in a Mesh.
 */ 
class SCICORESHARE MeshConcept : public Mesh
{
private:
  typedef unsigned int under_type;

public:

  //@{
  //! Index and Iterator types required for Mesh Concept.
  /*!
   *  These are pure types, they should have no data members or virtual
   *  functions associated with them, and thus no storage.  They are
   *  used to scope templates to the simplex types, and are never
   *  actually created or destroyed.  There are four of them that
   *  correspond to the first four simplices by dimensionality.
   *  Each symplex type has the following types associated with it
   *  
   *  index_type 
   * 	   A handle for a particular simplex in the geometry.
   * 	   
   *  iterator
   * 	   Forward iterator, supports the prefix ++ operator and dereferences
   * 	   to an index_type
   *  
   *  size_type
   * 	   Used to determine how many simplices there are in the geometry,
   * 	   and for resizing.
   *  
   *  array_type
   * 	   Used to return subsimplices, usually a vector<index_type>, but
   * 	   could be a fixed length vector for efficiency reasons.  It will
   * 	   still support the stl container types.
   *  
   *  In addition to these four types, a symplex type should also be
   *  supported by the get_type_description({Node, Edge, Face, Cell} *)
   *  function.
   */
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };					
  					
  struct Edge {				
    typedef EdgeIndex<under_type>       index_type;
    typedef EdgeIterator<under_type>    iterator;
    typedef EdgeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };					
					
  struct Face {				
    typedef FaceIndex<under_type>       index_type;
    typedef FaceIterator<under_type>    iterator;
    typedef FaceIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };					
					
  struct Cell {				
    typedef CellIndex<under_type>       index_type;
    typedef CellIterator<under_type>    iterator;
    typedef CellIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };
  //@}

  //! The element simplex type.
  /*! A convenience type, this points to the highest order simplex type
   *  for the mesh.  For example, in a tetrahedral mesh this would be
   *  an alias for Cell, in any surface mesh this would be an alias for
   *  Face, and in a PointCloudMesh this is an alias for Node.
   */
  typedef Edge Elem;

  //@{
  //! Basic constructors and destructors.
  /*! Any additional constructors belong here as well.  See LatVolMesh
   *  for examples.  The empty arguement constructor should probably be
   *  private until there more flexible ways to construct meshes.  The
   *  Destructor must be virtual in order for delete to work properly
   *  on MeshHandles.
   */
  MeshConcept();
  MeshConcept(const MeshConcept &copy);
  virtual MeshConcept *clone() { return scinew MeshConcept(*this); }
  virtual ~MeshConcept();
  //@}

  //@{
  //! Basic iterators.
  /*!
   *  TODO: Write what these iterators are good for here.
   *
   */
  void begin(Node::iterator &itr) const;
  void begin(Edge::iterator &itr) const;
  void begin(Face::iterator &itr) const;
  void begin(Cell::iterator &itr) const;

  void end(Node::iterator &itr) const;
  void end(Edge::iterator &itr) const;
  void end(Face::iterator &itr) const;
  void end(Cell::iterator &itr) const;

  void size(Node::size_type &size) const;
  void size(Edge::size_type &size) const;
  void size(Face::size_type &size) const;
  void size(Cell::size_type &size) const;
  //@}

  //! Returns the axis aligned bounding box of the mesh.
  virtual BBox get_bounding_box() const;
  //! Destructively applies the given transform to the mesh.
  virtual void transform(Transform &t);

  //@{
  //! Get the child elements of the given index.
  void get_nodes(Node::array_type &a, Edge::index_type i) const;
  void get_nodes(Node::array_type &a, Face::index_type i) const;
  void get_nodes(Node::array_type &a, Cell::index_type i) const;
  void get_edges(Edge::array_type &a, Face::index_type i) const;
  void get_edges(Edge::array_type &a, Cell::index_type i) const;
  void get_faces(Face::array_type &a, Cell::index_type i) const;
  //@}

  //@{
  //! Get the parent element(s) of the given index.
  unsigned get_edges(Edge::array_type &a, Node::index_type i) const;
  unsigned get_faces(Face::array_type &a, Node::index_type i) const;
  unsigned get_faces(Face::array_type &a, Edge::index_type i) const;
  unsigned get_cells(Cell::array_type &a, Node::index_type i) const;
  unsigned get_cells(Cell::array_type &a, Edge::index_type i) const;
  unsigned get_cells(Cell::array_type &a, Face::index_type i) const;
  //@}

  //! Get the neighbors of a simplex.  Similar to get_edges() with
  //Node::index_type argument, but returns the "other" edge if it
  //exists, not all that exist.
  void get_neighbor(Edge::index_type &, Node::index_type) const;

  //@{
  //! Get the center point of an element.
  void get_center(Point &result, Node::index_type idx) const;
  void get_center(Point &result, Edge::index_type idx) const;
  void get_center(Point &result, Face::index_type idx) const;
  void get_center(Point &result, Cell::index_type idx) const;
  //@}

  //@{
  //! Look up the index closest to or containing the given point.
  bool locate(Node::index_type &idx, const Point &p) const;
  bool locate(Edge::index_type &idx, const Point &p) const;
  bool locate(Face::index_type &idx, const Point &p) const;
  bool locate(Cell::index_type &idx, const Point &p) const;
  //@}

  //@{
  //! Compute the array of subsimplices and the interpolant weights
  //! around a given point.
  /*! These are used to compute the interpolation weights for a given
   *  point within the mesh.  Currently only the Node::array_type and
   *  the Elem::array_type specializations are implemented, the others
   *  do nothing for most meshes.
   */
  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Edge::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Face::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w);
  //@}

  //! Return true if the mesh is editable.
  /*! This method is optional.
   * If this is not defined the default return value is false.
   */
  virtual bool is_editable() const;
  //! Add a new point to the mesh.
  /*! This only works if the mesh is editable.
   */
  Node::index_type add_node(const Point &p);
  //! Add a new element to the mesh.
  /*! This only works if the mesh is editable.
   */
  Elem::index_type add_elem(Node::array_type a);
  //! Change a point within the mesh.
  /*! This only works if the mesh is editable.
   *  This method is optional.
   */
  void set_point(const Point &point, Node::index_type index);

  //! Flush any edits made to the mesh.
  /*! Any edits made to a mesh must be flushed with this call in order
   *  to guarantee that they show up in any mesh queries.  For instance,
   *  if a point is added to a PointCloudMesh, this must be called
   *  before the point is locatable.  Usually this call updates any
   *  search structures associated with a mesh.
   */
  virtual void flush_changes();
 

  //! Return true if the mesh has normals associated with the points.
  /*! This method is optional, and will return false if not defined. */
  virtual bool has_normals() const;

  //! Get a normal associated with the given node.
  /*! This method is optional, and only works if has_normals is true. */
  void get_normal(Vector &result, Node::index_type idx) const;

  //@{
  //! Support functions for the SCIRun Pio system.
  /*! These functions and definitions are used by the Pio system to read
   *  and write meshes.
   */
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  //@}

  //! Virtualized get_type_description().
  /*! get_type_description is used for dispatching on meshes if we
   *  don't already know the exact mesh type that we are using.  It is
   *  especially convenient for dynamically compiling code templated
   *  on mesh type.
   */
  virtual const TypeDescription *get_type_description() const;

private:

  //! Used by the PIO system to create a new mesh when reading in.
  static Persistent *maker();
};


typedef LockingHandle<MeshConcept> MeshConceptHandle;

const TypeDescription* get_type_description(MeshConcept *);
const TypeDescription* get_type_description(MeshConcept::Node *);
const TypeDescription* get_type_description(MeshConcept::Edge *);
const TypeDescription* get_type_description(MeshConcept::Face *);
const TypeDescription* get_type_description(MeshConcept::Cell *);


} // namespace SCIRun

#endif // SCI_project_MeshConcept_h





