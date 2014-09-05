/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef Datatypes_Mesh_h
#define Datatypes_Mesh_h

#include <sci_defs/hashmap_defs.h>

#include <Core/Math/MusilRNG.h>
#include <Core/Datatypes/PropertyManager.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/StackVector.h>
#include <Core/Datatypes/FieldVIndex.h>
#include <Core/Datatypes/FieldVIterator.h>

#include <Core/Datatypes/share.h>
#include <set>

namespace SCIRun {

class BBox;
class Mesh;
class Transform;
class TypeDescription;

typedef LockingHandle<Mesh> MeshHandle;

// Maximum number of weights get_weights will return.
// Currently at 15 for QuadraticLatVolMesh.
#define MESH_WEIGHT_MAXSIZE 64

// We reserve the last unsigned int value as a marker for bad mesh
// indices.  This is useful for example when there are no neighbors,
// or when elements are deleted.
#define MESH_INVALID_INDEX ((unsigned int)-1)
#define MESH_NO_NEIGHBOR MESH_INVALID_INDEX



class SCISHARE Mesh : public PropertyManager {
public:

  // VIRTUAL INTERFACE
  
  typedef unsigned int index_type;
  typedef unsigned int size_type;
  typedef std::vector<unsigned int> array_type;
  typedef StackVector<unsigned int,3> dimension_type;
  
  class VNode { 
    public:
      typedef VNodeIterator<Mesh::index_type>   iterator;
      typedef VNodeIndex<Mesh::index_type>      index_type;
      typedef VNodeIndex<Mesh::size_type>       size_type;
      typedef vector<index_type>                array_type;
  };

  class VEdge { 
    public:
      typedef VEdgeIterator<Mesh::index_type>   iterator;
      typedef VEdgeIndex<Mesh::index_type>      index_type;
      typedef VEdgeIndex<Mesh::size_type>       size_type;
      typedef vector<index_type>                array_type;
  };
  
  class VFace { 
    public:
      typedef VFaceIterator<Mesh::index_type>   iterator;    
      typedef VFaceIndex<Mesh::index_type>      index_type;
      typedef VFaceIndex<Mesh::size_type>       size_type;
      typedef vector<index_type>                array_type;
  };
  
  class VCell { 
    public:
      typedef VCellIterator<Mesh::index_type>   iterator;
      typedef VCellIndex<Mesh::index_type>      index_type;
      typedef VCellIndex<Mesh::size_type>       size_type;
      typedef vector<index_type>                array_type;
  };
  
  class VElem { 
    public:
      typedef VElemIterator<Mesh::index_type>   iterator;
      typedef VElemIndex<Mesh::index_type>      index_type;
      typedef VElemIndex<Mesh::size_type>       size_type;
      typedef vector<index_type>                array_type;
  };
  
  class VDElem { 
    public:
      typedef VDElemIterator<Mesh::index_type>  iterator;    
      typedef VDElemIndex<Mesh::index_type>     index_type;
      typedef VDElemIndex<Mesh::size_type>      size_type;
      typedef vector<index_type>                array_type;
  };
  
  
  //! Function to indicate whether function has been upgraded
  //! to support a virtual interface
  virtual bool has_virtual_interface() const;

  //! Get the number of elements in the mesh of the specified type
  //! Note: that for any size other then the number of nodes or
  //! elements, one has to synchronize that part of the mesh.
  virtual void size(VNode::size_type& size) const;
  virtual void size(VEdge::size_type& size) const;
  virtual void size(VFace::size_type& size) const;
  virtual void size(VCell::size_type& size) const;
  virtual void size(VElem::size_type& size) const;
  virtual void size(VDElem::size_type& size) const;
  
  //! Get the nodes that make up an element
  //! Depending on the geometry not every function may be available
  virtual void get_nodes(VNode::array_type& nodes, VNode::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VEdge::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VFace::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VCell::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VElem::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VDElem::index_type i) const;

  //! Get the edges that make up an element
  //! or get the edges that contain certain nodes
  //! Depending on the geometry not every function may be available
  virtual void get_edges(VEdge::array_type& edges, VNode::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VEdge::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VFace::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VCell::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VElem::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VDElem::index_type i) const;

  //! Get the faces that make up an element
  //! or get the faces that contain certain nodes or edges
  //! Depending on the geometry not every function may be available
  virtual void get_faces(VFace::array_type& faces, VNode::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VEdge::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VFace::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VCell::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VElem::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VDElem::index_type i) const;

  //! Get the cell index that contains the specified component
  //! Depending on the geometry not every function may be available
  virtual void get_cells(VCell::array_type& cells, VNode::index_type i) const;
  virtual void get_cells(VCell::array_type& cells, VEdge::index_type i) const;
  virtual void get_cells(VCell::array_type& cells, VFace::index_type i) const;
  virtual void get_cells(VCell::array_type& cells, VCell::index_type i) const;
  virtual void get_cells(VCell::array_type& cells, VElem::index_type i) const;
  virtual void get_cells(VCell::array_type& cells, VDElem::index_type i) const;

  virtual void get_elems(VElem::array_type& elems, VNode::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VEdge::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VFace::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VCell::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VElem::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VDElem::index_type i) const;

  virtual void get_delems(VDElem::array_type& delems, VNode::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VEdge::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VFace::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VCell::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VElem::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VDElem::index_type i) const;

  //! Get the center of a certain mesh element
  virtual void get_center(Point &point, VNode::index_type i) const;
  virtual void get_center(Point &point, VEdge::index_type i) const;
  virtual void get_center(Point &point, VFace::index_type i) const;
  virtual void get_center(Point &point, VCell::index_type i) const;
  virtual void get_center(Point &point, VElem::index_type i) const;
  virtual void get_center(Point &point, VDElem::index_type i) const;

  //! Get the geometrical sizes of the mesh elements
  virtual double get_size(VNode::index_type i) const;
  virtual double get_size(VEdge::index_type i) const;
  virtual double get_size(VFace::index_type i) const;
  virtual double get_size(VCell::index_type i) const;
  virtual double get_size(VElem::index_type i) const;
  virtual double get_size(VDElem::index_type i) const;
  
  virtual void get_weights(const Point& p,VNode::array_type& nodes,
                                                vector<double>& weights) const;
  virtual void get_weights(const Point& p,VElem::array_type& elems,
                                                vector<double>& weights) const;
  
  //! Locate where a position is in  the mesh
  //! The node version finds the closest node
  //! The element version find the element that contains the point
  virtual bool locate(VNode::index_type &i, const Point &point) const;
  virtual bool locate(VElem::index_type &i, const Point &point) const;

  //! Find the coordinates of a point in a certain element
  virtual bool get_coords(vector<double> &coords, const Point &point, 
                                                    VElem::index_type i) const;
  
  //! Interpolate from local coordinates to global coordinates
  virtual void interpolate(Point &p, const vector<double> &coords, 
                                                    VElem::index_type i) const;

  //! Interpolate from local coordinates to a derivative in local coordinates  
  virtual void derivate(vector<Point> &p, const vector<double> &coords, 
                                                    VElem::index_type i) const;
  
  virtual void get_normal(Vector &result, vector<double> &coords, 
                                 VElem::index_type eidx, unsigned int f) const;
  
  //! Set and get a node location.
  //! Node set is only available for editable meshes
  virtual void get_points(vector<Point>& points) const;
  
  virtual void get_random_point(Point &p, VElem::index_type i,MusilRNG &rng) const;
  virtual void set_point(const Point &point, VNode::index_type i);
  
  //! Preallocate memory for better performance
  virtual void node_reserve(size_t size);
  virtual void elem_reserve(size_t size);

  //! Add a node to a mesh
  virtual void add_node(const Point &point,VNode::index_type &i);
  
  //! Add an element to a mesh
  virtual void add_elem(const VNode::array_type &nodes,VElem::index_type &i);

  //! Get the neighbors of a node or an element
  virtual bool get_neighbor(VElem::index_type &neighbor, 
                        VElem::index_type from, VDElem::index_type delem) const;
  virtual void get_neighbors(VElem::array_type &elems, 
                                                    VElem::index_type i) const;
  virtual void get_neighbors(VNode::array_type &nodes, 
                                                    VNode::index_type i) const;

  //! Draw non linear elements
  virtual void pwl_approx_edge(vector<vector<double> > &coords, 
                               VElem::index_type ci, unsigned int which_edge, 
                               unsigned int div_per_unit) const;
  virtual void pwl_approx_face(vector<vector<vector<double> > > &coords, 
                               VElem::index_type ci, unsigned int which_face, 
                               unsigned int div_per_unit) const;

  //! Get node normals, needed for visualization
  virtual void get_normal(Vector& norm,VNode::index_type i) const;

  //! Get the basis order of the mesh (implemented in every mesh)
  virtual int basis_order();

  //! Get the dimensions of the mesh.
  //! This function will replace get_dim()
  virtual void get_dimensions(dimension_type& dim);

  //----------------------------------------------------------------------
  // Functions that are based on the virtual ones
  // These functions recreate functions we used in dynamic compilation:


  inline void get_point(Point &point, VNode::index_type i) const
  { get_center(point,i); }
  
  inline VNode::index_type add_node(Point &point)
  { VNode::index_type i; add_node(point,i); return (i); }

  inline double get_length(VEdge::index_type i) const
  { return (get_size(i)); }

  inline double get_area(VFace::index_type i) const
  { return (get_size(i)); }
  
  inline double get_volume(VCell::index_type i) const
  { return (get_size(i)); }
  
  inline size_t num_nodes() const
  {
    VNode::index_type s;
    size(s);
    return(static_cast<size_t>(s));
  }

  inline size_t num_edges() const
  {
    VEdge::index_type s;
    size(s);
    return(static_cast<size_t>(s));
  }
  
  inline size_t num_faces() const
  {
    VFace::index_type s;
    size(s);
    return(static_cast<size_t>(s));
  }

  inline size_t num_cells() const
  {
    VCell::index_type s;
    size(s);
    return(static_cast<size_t>(s));
  }
  
  inline size_t num_elems() const
  {
    VElem::index_type s;
    size(s);
    return(static_cast<size_t>(s));
  }

  inline size_t num_delems() const
  {
    VDElem::index_type s;
    size(s);
    return(static_cast<size_t>(s));
  }
    
  inline void begin(VNode::iterator &it) const
  {
    it = 0;
  }

  inline void begin(VEdge::iterator &it) const
  {
    it = 0;
  }

  inline void begin(VFace::iterator &it) const
  {
    it = 0;
  }

  inline void begin(VCell::iterator &it) const
  {
    it = 0;
  }

  inline void begin(VElem::iterator &it) const
  {
    it = 0;
  }

  inline void begin(VDElem::iterator &it) const
  {
    it = 0;
  }

  inline void end(VNode::iterator &it) const
  {
    VNode::size_type s;
    size(s);
    it = static_cast<index_type>(s);
  }

  inline void end(VEdge::iterator &it) const
  {
    VEdge::size_type s;
    size(s);
    it = static_cast<index_type>(s);
  }

  inline void end(VFace::iterator &it) const
  {
    VFace::size_type s;
    size(s);
    it = static_cast<index_type>(s);
  }

  inline void end(VCell::iterator &it) const
  {
    VCell::size_type s;
    size(s);
    it = static_cast<index_type>(s);
  }
  
  inline void end(VElem::iterator &it) const
  {
    VElem::size_type s;
    size(s);
    it = static_cast<index_type>(s);
  }

  inline void end(VDElem::iterator &it) const
  {
    VDElem::size_type s;
    size(s);
    it = static_cast<index_type>(s);
  }
  
  inline Mesh::VNode::index_type add_node(const Point& point) 
  {
    VNode::index_type idx;
    add_node(point,idx);
    return (idx);
  }

  inline Mesh::VNode::index_type add_point(const Point& point) 
  {
    VNode::index_type idx;
    add_node(point,idx);
    return (idx);
  }
  
  inline Mesh::VElem::index_type add_elem(const VNode::array_type nodes)
  {
    VElem::index_type idx;
    add_elem(nodes,idx);
    return (idx);
  }

  inline Mesh::VEdge::index_type add_edge(const VNode::array_type nodes)
  {
    VElem::index_type idx;
    add_elem(nodes,idx);
    return (VEdge::index_type(idx));
  }

  inline Mesh::VFace::index_type add_tri(const VNode::array_type nodes)
  {
    VElem::index_type idx;
    add_elem(nodes,idx);
    return (VFace::index_type(idx));
  }

  inline Mesh::VFace::index_type add_quad(const VNode::array_type nodes)
  {
    VElem::index_type idx;
    add_elem(nodes,idx);
    return (VFace::index_type(idx));
  }

  inline Mesh::VCell::index_type add_tet(const VNode::array_type nodes)
  {
    VElem::index_type idx;
    add_elem(nodes,idx);
    return (VCell::index_type(idx));
  }

  inline Mesh::VCell::index_type add_prism(const VNode::array_type nodes)
  {
    VElem::index_type idx;
    add_elem(nodes,idx);
    return (VCell::index_type(idx));
  }

  inline Mesh::VCell::index_type add_hex(const VNode::array_type nodes)
  {
    VElem::index_type idx;
    add_elem(nodes,idx);
    return (VCell::index_type(idx));
  }

  
  template <class VEC1, class VEC2>
  inline void convert_vector(VEC1& v1, VEC2 v2) const
  {
    v1.resize(v2.size());
    for (size_t p=0; p < v2.size(); p++) v1[p] = static_cast<typename VEC1::value_type>(v2[p]);
  }
  
  virtual Mesh *clone() = 0;

  Mesh();
  virtual ~Mesh();
  
  //! Required virtual functions.
  virtual BBox get_bounding_box() const = 0;

  //! Destructively applies the given transform to the mesh.
  virtual void transform(const Transform &t) = 0;

  //! Return the transformation that takes a 0-1 space bounding box 
  //! to the current bounding box of this mesh.
  virtual void get_canonical_transform(Transform &t);

  enum
  { 
    UNKNOWN		= 0,
    STRUCTURED		= 1 << 1,
    UNSTRUCTURED	= 1 << 2,
    REGULAR		= 1 << 3,
    IRREGULAR   	= 1 << 4
  };

  enum
  { 
    NONE_E		= 0,
    NODES_E		= 1 << 0,
    EDGES_E		= 1 << 1,
    FACES_E		= 1 << 2,
    CELLS_E		= 1 << 3,
    ALL_ELEMENTS_E      = NODES_E | EDGES_E | FACES_E | CELLS_E,
    NORMALS_E		= 1 << 4,
    NODE_NEIGHBORS_E	= 1 << 5,
    EDGE_NEIGHBORS_E	= 1 << 6,
    FACE_NEIGHBORS_E	= 1 << 7,
    LOCATE_E		= 1 << 8,
    ELEMS_E  = 1 << 9,
    DELEMS_E = 1 << 10,
    ELEM_NEIGHBORS_E = 1 << 11,
    DELEM_NEIGHBORS_E = 1 << 12
  };
  virtual bool		synchronize(unsigned int) { return false; };


  //! Optional virtual functions.
  //! finish all computed data within the mesh.
  virtual bool has_normals() const { return false; }
  virtual bool is_editable() const { return false; } // supports add_elem(...)
  virtual int  dimensionality() const = 0;
  virtual int  topology_geometry() const = 0;
  virtual bool get_dim(vector<unsigned int>&) const { return false;  }
  virtual bool get_search_grid_info(int &i, int &j, int &k, Transform &trans) { return false; }
  // Required interfaces
  
  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const = 0;

  // The minimum value for elemental checking
  double MIN_ELEMENT_VAL;
};

//! General case locate, search each elem.
template <class Msh>
bool elem_locate(typename Msh::Elem::index_type &elem,
		 Msh &msh, const Point &p) 
{
  typename Msh::Elem::iterator iter, end;
  msh.begin(iter);
  msh.end(end);
  vector<double> coords(msh.dimensionality());
  while (iter != end) {
    if (msh.get_coords(coords, p, *iter)) {
      elem = *iter;
      return true;
    }
    ++iter;
  }
  return false;
}


typedef LockingHandle<Mesh> MeshHandle;

class SCISHARE MeshTypeID {
  public:
    // Constructor
    MeshTypeID(const string& type, 
                MeshHandle (*mesh_maker)());
    MeshTypeID(const string& type, 
                MeshHandle (*mesh_maker)(),
                MeshHandle (*latvol_maker)(unsigned int x, unsigned int y, unsigned int z, const Point& min, const Point& max)
                );
    MeshTypeID(const string& type, 
                MeshHandle (*mesh_maker)(),
                MeshHandle (*image_maker)(unsigned int x, unsigned int y, const Point& min, const Point& max)
                );
    MeshTypeID(const string& type, 
                MeshHandle (*mesh_maker)(),
                MeshHandle (*scanline_maker)(unsigned int x,const Point& min, const Point& max)
                );
    MeshTypeID(const string& type, 
                MeshHandle (*mesh_maker)(),
                MeshHandle (*structhexvol_maker)(unsigned int x, unsigned int y, unsigned int z)
                );
    MeshTypeID(const string& type, 
                MeshHandle (*mesh_maker)(),
                MeshHandle (*structquadsurf_maker)(unsigned int x, unsigned int y)
                );
    MeshTypeID(const string& type, 
                MeshHandle (*mesh_maker)(),
                MeshHandle (*structcurve_maker)(unsigned int x)
                );

    
    string type;
    MeshHandle (*mesh_maker)();
    
    // Custom Constructors
    MeshHandle (*latvol_maker)(unsigned int x, unsigned int y, unsigned int z, const Point& min, const Point& max);
    MeshHandle (*image_maker)(unsigned int x, unsigned int y, const Point& min, const Point& max);
    MeshHandle (*scanline_maker)(unsigned int x, const Point& min, const Point& max);
    MeshHandle (*structhexvol_maker)(unsigned int x, unsigned int y, unsigned int z);
    MeshHandle (*structquadsurf_maker)(unsigned int x, unsigned int y);
    MeshHandle (*structcurve_maker)(unsigned int x);

    
};


MeshHandle Create_Mesh(string type);
MeshHandle Create_Mesh(string type,unsigned int x, unsigned int y, unsigned int z, const Point& min, const Point& max);
MeshHandle Create_Mesh(string type,unsigned int x, unsigned int y, const Point& min, const Point& max);
MeshHandle Create_Mesh(string type,unsigned int x, const Point& min, const Point& max);
MeshHandle Create_Mesh(string type,unsigned int x, unsigned int y, unsigned int z);
MeshHandle Create_Mesh(string type,unsigned int x, unsigned int y);
MeshHandle Create_Mesh(string type,unsigned int x);

} // end namespace SCIRun

#endif // Datatypes_Mesh_h
