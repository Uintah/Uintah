/*
 *  LatVolMesh.h: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef SCI_project_LatVolMesh_h
#define SCI_project_LatVolMesh_h 1

#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/MeshBase.h>


namespace SCIRun {


class SCICORESHARE LatVolMesh : public MeshBase
{
public:
  struct LatIndex
  {
    LatIndex() : i_(0), j_(0), k_(0) {}
    LatIndex(unsigned i, unsigned j, unsigned k) : i_(i), j_(j), k_(k) {}
    
    unsigned i_, j_, k_;
  };

  struct CellIndex : public LatIndex
  {
    CellIndex() : LatIndex() {}
    CellIndex(unsigned i, unsigned j, unsigned k) : LatIndex(i,j,k) {}    
  };
  
  struct NodeIndex : public LatIndex
  {
    NodeIndex() : LatIndex() {}
    NodeIndex(unsigned i, unsigned j, unsigned k) : LatIndex(i,j,k) {}    
  };
  
  struct NCIter : public LatIndex
  {
    NCIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : LatIndex(i, j, k), mesh_(m) {}
    
    const LatIndex &operator *() { return *this; }
    
    bool operator ==(const NCIter &a)
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }
    
    bool operator !=(const NCIter &a)
    {
      return !(*this == a);
    }
    
    const LatVolMesh *mesh_;
  };
  
  
  struct NodeIter : public NCIter
  {
    NodeIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : NCIter(m, i, j, k) {}
    
    NodeIter &operator++()
    {
      i_++;
      if (i_ > mesh_->nx_)
	{
	  i_ = 0;
	  j_++;
	  if (j_ > mesh_->ny_)
	    {
	      j_ = 0;
	      k_++;
	    }
	}
      return *this;
    }
    
    NodeIter operator++(unsigned)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }
  };
  
  
  struct CellIter : public NCIter
  {
    CellIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k) : NCIter(m, i, j, k) {}
    
    CellIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->nx_)
	{
	  i_ = 0;
	  j_++;
	  if (j_ >= mesh_->ny_)
	    {
	      j_ = 0;
	      k_++;
	    }
	}
      return *this;
    }
    
  private:
    CellIter operator++(unsigned)
    {
      CellIter result(*this);
      operator++();
      return result;
    }
  };
  
  typedef LatIndex index_type;
  
  //! Index and Iterator types required for Mesh Concept.
  typedef NodeIndex       node_index;
  typedef NodeIter        node_iterator;
 
  //typedef EdgeIndex       edge_index;     
  //typedef EdgeIterator    edge_iterator;
 
  //typedef FaceIndex       face_index;
  //typedef FaceIterator    face_iterator;
 
  typedef CellIndex       cell_index;
  typedef CellIter        cell_iterator;

  //! Storage types for the arguments passed to the 
  //  get_*() functions.  For lattice meshes, these all have
  //  known maximum sizes, so we use them.
  typedef node_index  node_array[8];
  typedef edge_index  edge_array[12];
  typedef face_index  face_array[6];
  typedef cell_index  cell_array[8];

  LatVolMesh(unsigned x, unsigned y, unsigned z, Point &min, Point &max);
  LatVolMesh(const LatVolMesh &);
  virtual ~LatVolMesh();

  node_iterator node_begin() const;
  node_iterator node_end() const;
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  face_iterator face_begin() const;
  face_iterator face_end() const;
  cell_iterator cell_begin() const;
  cell_iterator cell_end() const;

  //! get the mesh statistics
  unsigned get_nx() const { return nx_; }
  unsigned get_ny() const { return ny_; }
  unsigned get_nz() const { return nz_; }
  Point get_min() const { return min_; }
  Point get_max() const { return max_; }

  virtual BBox get_bounding_box() const;

  //! get the child elements of the given index
  void get_nodes(node_array &array, edge_index idx) const;
  void get_nodes(node_array &array, face_index idx) const;
  void get_nodes(node_array &array, cell_index idx) const;
  void get_edges(edge_array &array, face_index idx) const;
  void get_edges(edge_array &array, cell_index idx) const;
  void get_faces(face_array &array, cell_index idx) const;

  //! get the parent element(s) of the given index
  unsigned get_edges(edge_array &array, node_index idx) const;
  unsigned get_faces(face_array &array, node_index idx) const;
  unsigned get_faces(face_array &array, edge_index idx) const;
  unsigned get_cells(cell_array &array, node_index idx) const;
  unsigned get_cells(cell_array &array, edge_index idx) const;
  unsigned get_cells(cell_array &array, face_index idx) const;

  //! similar to get_cells() with face_index argument, but
  //  returns the "other" cell if it exists, not all that exist
  void get_neighbor(cell_index &neighbor, face_index idx) const;

  //! get the center point (in object space) of an element
  void get_center(Point &result, node_index idx) const;
  void get_center(Point &result, edge_index idx) const;
  void get_center(Point &result, face_index idx) const;
  void get_center(Point &result, cell_index idx) const;

  void locate_node(node_index &node, const Point &p) const;
  void locate_edge(edge_index &edge, const Point &p, double[2]) const;
  void locate_face(face_index &face, const Point &p, double[4]) const;
  void locate_cell(cell_index &cell, const Point &p, double[8]) const;


  void unlocate(Point &result, const Point &p) const;

  void get_point(Point &result, const node_index &index) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) const { return type_name(n); }


private:

  //! the object space extents of a LatVolMesh
  Point min_, max_;

  //! the node_index extents of a LatVolMesh (min=0, max=n)
  unsigned nx_, ny_, nz_;

  // returns a LatVolMesh
  static Persistent *maker();
};



} // namespace SCIRun

#endif // SCI_project_LatVolMesh_h





