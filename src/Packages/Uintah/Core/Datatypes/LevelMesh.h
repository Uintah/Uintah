/*
 *  LevelMesh.h: Templated Mesh defined on a 3D Regular Grid
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

#ifndef SCI_project_LevelMesh_h
#define SCI_project_LevelMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <string>
#include <vector>

using SCIRun::Point;
using SCIRun::LockingHandle;
using SCIRun::MeshBase;
using SCIRun::IntVector;
using SCIRun::BBox;
using SCIRun::Piostream;
using SCIRun::PersistentTypeID;
using SCIRun::Persistent;

using std::string;
using std::vector;

namespace Uintah {

class LevelMesh;
typedef LockingHandle<LevelMesh> LevelMeshHandle;
  
class SCICORESHARE LevelMesh : public MeshBase
{
public:
  struct LevelIndex
  {
    public:
    LevelIndex(const LevelMesh* m, int i, int j, int k);
    const LevelMesh *mesh_;
    int i_, j_, k_;
    const Patch* patch_;
    protected:
    LevelIndex(){}   
  };
  
  struct CellIndex : public LevelIndex
  {
    CellIndex() : LevelIndex() {}
    CellIndex(const LevelMesh *m, int i, int j, int k);
  };
  
  struct NodeIndex : public LevelIndex
  {
    NodeIndex() : LevelIndex() {}
    NodeIndex(const LevelMesh *m, int i, int j, int k):
      LevelIndex(m, i, j, k) {}
  };
  
  struct LevelIter : public LevelIndex
  {
    LevelIter(const LevelMesh *m,int i, int j, int k ) :
      LevelIndex(m,i,j,k) {}
    
    const LevelIndex &operator *() { return *this; }
    
    bool operator ==(const LevelIter &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_
	&& patch_ == a.patch_;
    }
    
    bool operator !=(const LevelIter &a) const
    {
      return !(*this == a);
    }
  };

  struct NodeIter : public LevelIter
  {
    NodeIter(const LevelMesh *m, int i, int j, int k) 
      : LevelIter(m, i, j, k) {}
    
    const NodeIndex &operator *() const { return (const NodeIndex&)(*this); }

    NodeIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->idxLow_.x() + mesh_->nx_)	{
	i_ = mesh_->idxLow_.x();
	j_++;
	if (j_ >= mesh_->idxLow_.y() +mesh_->ny_) {
	  j_ = mesh_->idxLow_.y();
	  k_++;
	}
      }
      IntVector idx = IntVector( i_, j_, k_);
      if( !(patch_->containsNode( idx ) ) )
	patch_ = mesh_->grid_->getLevel(mesh_->level_)->
	  selectPatchForNodeIndex( idx ); 
      return *this;
    }
    
  private:

    NodeIter operator++(int)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }
  };
  
  
  struct CellIter : public LevelIter
  {
    CellIter(const LevelMesh *m, int i, int j, int k)
      : LevelIter(m, i, j, k) {}
    
    const CellIndex &operator *() const { return (const CellIndex&)(*this); }

    CellIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->idxLow_.x() +mesh_->nx_-1) {
	i_ = mesh_->idxLow_.x();
	j_++;
	if (j_ >= mesh_->idxLow_.y() + mesh_->ny_-1) {
	  j_ = mesh_->idxLow_.y();
	  k_++;
	}
      }
      IntVector idx = IntVector( i_, j_, k_);
      if( !(patch_->containsNode( idx ) ))
	patch_ = mesh_->grid_->getLevel(mesh_->level_)-> 
	  selectPatchForCellIndex( idx ); 
      return *this;
    }
    
  private:

    CellIter operator++(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }
  };
  
  struct UnfinishedIndex
  {
  public:
    UnfinishedIndex() : i_(0) {}
    UnfinishedIndex(int i) : i_(i) {}

    operator const int() const { return i_; }

    int i_;
  };

  struct EdgeIndex : public UnfinishedIndex
  {
    EdgeIndex() : UnfinishedIndex() {}
    EdgeIndex(int i) : UnfinishedIndex(i) {}
  };

  struct FaceIndex : public UnfinishedIndex
  {
    FaceIndex() : UnfinishedIndex() {}
    FaceIndex(int i) : UnfinishedIndex(i) {}
  };

  struct UnfinishedIter : public UnfinishedIndex
  {
    UnfinishedIter(const LevelMesh *m, int i) 
      : UnfinishedIndex(i), mesh_(m) {}
    
    const UnfinishedIndex &operator *() { return *this; }
    
    bool operator ==(const UnfinishedIter &a) const
    {
      return i_ == a.i_ && mesh_ == a.mesh_;
    }
    
    bool operator !=(const UnfinishedIter &a) const
    {
      return !(*this == a);
    }
    
    const LevelMesh *mesh_;
  };

  struct EdgeIter : public UnfinishedIter
  {
    EdgeIter(const LevelMesh *m, int i) 
      : UnfinishedIter(m, i) {}
    
    const EdgeIndex &operator *() const { return (const EdgeIndex&)(*this); }

    EdgeIter &operator++() { return *this; }

  private:

    EdgeIter operator++(int)
    {
      EdgeIter result(*this);
      operator++();
      return result;
    }
  };

  struct FaceIter : public UnfinishedIter
  {
    FaceIter(const LevelMesh *m, int i) 
      : UnfinishedIter(m, i) {}
    
    const FaceIndex &operator *() const { return (const FaceIndex&)(*this); }

    FaceIter &operator++() { return *this; }

  private:

    FaceIter operator++(int)
    {
      FaceIter result(*this);
      operator++();
      return result;
    }
  };  

  typedef LevelIndex index_type;
  
  //! Index and Iterator types required for Mesh Concept.
  typedef NodeIndex       node_index;
  typedef NodeIter        node_iterator;
  typedef NodeIndex       node_size_type;

  typedef EdgeIndex       edge_index;     
  typedef EdgeIter        edge_iterator;
  typedef EdgeIndex       edge_size_type;
 
  typedef FaceIndex       face_index;
  typedef FaceIter        face_iterator;
  typedef FaceIndex       face_size_type;
 
  typedef CellIndex       cell_index;
  typedef CellIter        cell_iterator;
  typedef CellIndex       cell_size_type;
  // storage types for get_* functions
  typedef vector<node_index>  node_array;
  typedef vector<edge_index>  edge_array;
  typedef vector<face_index>  face_array;
  typedef vector<cell_index>  cell_array;
  typedef vector<double>      weight_array;

  friend class NodeIter;
  friend class CellIter;
  friend class EdgeIter;
  friend class FaceIter;
  friend class LevelIter;
  friend class LevelIndex;
  friend class NodeIndex;
  friend class CellIndex;

  // For now we must create a Level Mesh for each level of the Level
  LevelMesh() : grid_(0), level_(0) {}
  // if the above constructor is used a Grid and level must be added before
  // we can use the LevelMesh;
  void SetMesh( GridP g, int l){ grid_ = g; level_ = l; init(); }
  void init();
  // remaining constructors
  LevelMesh( GridP  g, int l) : grid_(g), level_(l) { init();}
  LevelMesh( LevelMesh* mh, int mx, int my, int mz,
	     int x, int y, int z); 
  LevelMesh(const LevelMesh &copy) :
    grid_(copy.grid_), level_(copy.level_), idxLow_(copy.idxLow_),
    nx_(copy.nx_), ny_(copy.ny_), nz_(copy.nz_), min_(copy.min_),
    max_(copy.max_) {}
  virtual LevelMesh *clone(){ return scinew LevelMesh(*this); }
  virtual ~LevelMesh() {}

  node_index  node(int i, int j, int k) const
    { return node_index(this, i, j, k); }
  node_iterator node_begin() const { return node_iterator(this, 0, 0, 0); }
  node_iterator node_end() const { return node_iterator(this,0, 0, nz_); }
  node_size_type nodes_size() const {
    return node_size_type(this, nx_, ny_, nz_);
  }
  edge_iterator edge_begin() const { return edge_iterator(this, 0); }
  edge_iterator edge_end() const  { return edge_iterator(this, 0); }
  edge_size_type edges_size() const { return edge_size_type(0); }

  face_iterator face_begin() const { return face_iterator(this,0); }
  face_iterator face_end() const { return face_iterator(this,0); }
  face_size_type faces_size() const { return face_size_type(0); }

  cell_index  cell(int i, int j, int k) const
    { return cell_index(this, i, j, k); }
   cell_iterator cell_begin() const { return cell_iterator(this, 0, 0, 0); }
  cell_iterator cell_end() const { return cell_iterator(this, 0, 0, nz_-1); }
  cell_size_type cells_size() const {
    return cell_size_type(this, nx_-1, ny_-1, nz_-1);
  }

  //! get the mesh statistics
  int get_nx() const { return nx_; }
  int get_ny() const { return ny_; }
  int get_nz() const { return nz_; }
  Point get_min() const { return min_; }
  Point get_max() const { return max_; }
  Vector diagonal() const { return max_ - min_; }
  virtual BBox get_bounding_box() const;

  //! set the mesh statistics
  void set_nx(int x) { nx_ = x; }
  void set_ny(int y) { ny_ = y; }
  void set_nz(int z) { nz_ = z; }
  void set_min(Point p) { min_ = p; }
  void set_max(Point p) { max_ = p; }

  //! get the child elements of the given index
  void get_nodes(node_array &, edge_index) const {}
  void get_nodes(node_array &, face_index) const {}
  void get_nodes(node_array &array, cell_index idx) const;
  void get_edges(edge_array &, face_index) const {}
  void get_edges(edge_array &, cell_index) const {}
  void get_faces(face_array &, cell_index) const {}

  //! get the parent element(s) of the given index
  int get_edges(edge_array &, node_index) const { return 0; }
  int get_faces(face_array &, node_index) const { return 0; }
  int get_faces(face_array &, edge_index) const { return 0; }
  int get_cells(cell_array &, node_index) const { return 0; }
  int get_cells(cell_array &, edge_index) const { return 0; }
  int get_cells(cell_array &, face_index) const { return 0; }

  //! similar to get_cells() with face_index argument, but
  //  returns the "other" cell if it exists, not all that exist
  void get_neighbor(cell_index &, face_index) const {}

  //! get the center point (in object space) of an element
  void get_center(Point &result, node_index idx) const;
  void get_center(Point &, edge_index) const {}
  void get_center(Point &, face_index) const {}
  void get_center(Point &result, cell_index idx) const;


  bool locate(node_index &node, const Point &p) const;
  bool locate(edge_index &, const Point &) const { return false; }
  bool locate(face_index &, const Point &) const { return false; }
  bool locate(cell_index &cell, const Point &p) const;

  void unlocate(Point &result, const Point &p) const { result =  p; };

  void get_point(Point &result, node_index index) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:

  // each LevelMesh needs grid and level index
  GridP grid_; 
  int level_;
  IntVector idxLow_; // cache the low index


  //! the node_index space extents of a LevelMesh (min=0, max=n-1)
  int nx_, ny_, nz_;

  //! the object space extents of a LevelMesh
  Point min_, max_;

  // returns a LevelMesh
  static Persistent *maker() { return scinew LevelMesh(); }
};




} // namespace SCIRun

#endif // SCI_project_LevelMesh_h

