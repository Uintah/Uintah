/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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

//! The Mesh Concept, a sample mesh with all the required interface.
/*! This is all the things that belong in a Mesh.
 */ 
class MeshConcept : public Mesh
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
    typedef std::vector<index_type>     array_type;
  };					
  					
  struct Edge {				
    typedef EdgeIndex<under_type>       index_type;
    typedef EdgeIterator<under_type>    iterator;
    typedef EdgeIndex<under_type>       size_type;
    typedef std::vector<index_type>     array_type;
  };					
					
  struct Face {				
    typedef FaceIndex<under_type>       index_type;
    typedef FaceIterator<under_type>    iterator;
    typedef FaceIndex<under_type>       size_type;
    typedef std::vector<index_type>     array_type;
  };					
					
  struct Cell {				
    typedef CellIndex<under_type>       index_type;
    typedef CellIterator<under_type>    iterator;
    typedef CellIndex<under_type>       size_type;
    typedef std::vector<index_type>     array_type;
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

  //! for meshes that dimensionality makes sense (structured) fill the vector
  //! with the size in each dimension for all of the axes
  //! unstructured meshses will return false, and not touch the vector.
  bool get_dim(std::vector<unsigned int>&) const { return false;  }

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
  virtual void transform(const Transform &t);
  //! Return the transformation that takes a 0-1 space bounding box 
  //! to the current bounding box of this mesh.
  virtual bool get_canonical_transform(Transform &t);

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


  //@{
  //! Get the nieghboring elements of the same simplex type
  //! Note: Depending on the mesh type, this Can have multiple definitions, 
  //! so make sure to document exactly what is returned
  void get_neighbors(Node::array_type &, Node::index_type) const {};
  void get_neighbors(Edge::array_type &, Edge::index_type) const {};
  void get_neighbors(Face::array_type &, Face::index_type) const {};
  void get_neighbors(Cell::array_type &, Cell::index_type) const {};
  //@}


  //@{
  //! Get the center point of an element.
  void get_center(Point &result, Node::index_type idx) const;
  void get_center(Point &result, Edge::index_type idx) const;
  void get_center(Point &result, Face::index_type idx) const;
  void get_center(Point &result, Cell::index_type idx) const;
  //@}

  //@{
  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type idx) const;
  double get_size(Edge::index_type idx) const;
  double get_size(Face::index_type idx) const;
  double get_size(Cell::index_type idx) const;
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };
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
   *  point within the mesh.  The weights will be returned in w, and
   *  there will be at most MESH_WEIGHT_MAXSIZE of them (w should be of
   *  type double[MESH_WEIGHT_MAXSIZE].  The number of weights computed is
   *  returned by the function.  Currently only the Node::array_type
   *  and the Elem::array_type specializations are implemented, the
   *  others do nothing for most meshes.
   */
  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point &p, Edge::array_type &l, double *w);
  int get_weights(const Point &p, Face::array_type &l, double *w);
  int get_weights(const Point &p, Cell::array_type &l, double *w);
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
   *  This also works on non-editable meshes that are structured.
   *  This method is optional.
   */
  void set_point(const Point &point, Node::index_type index);

  //! Reserve space in the mesh for new elements.
  /*! This only works if the mesh is editable.  This is an
   *  optimization and may do nothing.  Elem reserve reserves the
   *  number of new elements to be added, not their size.  For example
   *  if you are going to add 8 tris to a mesh you should reserve 8
   *  and not 24.
   */
  void node_reserve(size_t s);
  void elem_reserve(size_t s);

  //! Flush any edits made to the mesh.
  /*! Any edits made to a mesh must be flushed with this call in order
   *  to guarantee that they show up in any mesh queries.  For instance,
   *  if a point is added to a PointCloudMesh, this must be called
   *  before the point is locatable.  Usually this call updates any
   *  search structures associated with a mesh.  */
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
  static  const std::string type_name(int n = -1);
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





