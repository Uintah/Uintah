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
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <string>
#include <vector>
#include <iostream>
using std::cerr;
using std::endl;

using SCIRun::Point;
using SCIRun::LockingHandle;
using SCIRun::Mesh;
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
  
class SCICORESHARE LevelMesh : public Mesh
{
public:
  struct LevelIndex
  {
    public:
    LevelIndex(const LevelMesh* m, int i, int j, int k);
    const LevelMesh *mesh_;
    int i_, j_, k_;
    const Patch* patch_;
    operator unsigned() const { return (unsigned)(i_*j_*k_); }
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
    NodeIter() : LevelIter(0, 0, 0, 0) {}
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
    CellIter() : LevelIter(0, 0, 0, 0) {}
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

    operator int() const { return i_; }

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
    EdgeIter() : UnfinishedIter(0, 0) {}
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
    FaceIter() : UnfinishedIter(0, 0) {}
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

  //typedef LevelIndex under_type;
  
  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex          index_type;
    typedef NodeIter           iterator;
    typedef NodeIndex          size_type;
    typedef vector<index_type> array_type;
  };			    
			    
  struct Edge {		    
    typedef EdgeIndex          index_type;     
    typedef EdgeIter           iterator;
    typedef EdgeIndex          size_type;
    typedef vector<index_type> array_type;
  };			       
			       
  struct Face {		       
    typedef FaceIndex          index_type;
    typedef FaceIter           iterator;
    typedef FaceIndex          size_type;
    typedef vector<index_type> array_type;
  };			       
			       
  struct Cell {		       
    typedef CellIndex          index_type;
    typedef CellIter           iterator;
    typedef CellIndex          size_type;
    typedef vector<index_type> array_type;
  };

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
  virtual void transform( Transform &t );
  virtual LevelMesh *clone(){ return scinew LevelMesh(*this); }
  virtual ~LevelMesh() {cerr<<"Deleting Level Mesh\n";}

  Node::index_type node(int i, int j, int k)
  { return Node::index_type(this, i, j, k); }

  void begin(Node::iterator &itr) const { itr=Node::iterator(this, 0, 0, 0); }
  void end(Node::iterator &itr) const { itr=Node::iterator(this,0, 0, nz_); }
  void size(Node::size_type &s) const { s=Node::size_type(this, nx_, ny_, nz_); }

  void begin(Edge::iterator &itr) const { itr = Edge::iterator(this, 0); }
  void end(Edge::iterator &itr) const  { itr = Edge::iterator(this, 0); }
  void size(Edge::size_type &s) const { s = Edge::size_type(0); }

  void begin(Face::iterator &itr) const { itr = Face::iterator(this,0); }
  void end(Face::iterator &itr) const { itr = Face::iterator(this,0); }
  void size(Face::size_type &s) const { s = Face::size_type(0); }

  void begin(Cell::iterator &itr) const {itr=Cell::iterator(this, 0, 0, 0); }
  void end(Cell::iterator &itr) const {itr=Cell::iterator(this, 0, 0, nz_-1); }
  void size(Cell::size_type &s) const {
    cerr<<"in size(Cell) with "<<nx_-1<<","<< ny_-1<<","<< nz_-1<<endl;
    s = Cell::size_type(this, nx_-1, ny_-1, nz_-1);
  }

  //! get the mesh statistics
  int get_nx() const { return nx_; }
  int get_ny() const { return ny_; }
  int get_nz() const { return nz_; }
  Point get_min() const { return min_; }
  Point get_max() const { return max_; }
  // Unsafe due to non-constness of unproject.
  Transform &get_transform(){ return transform_;}
  Transform &set_transform(const Transform &trans) {
    transform_ = trans; return transform_;}
  
  Vector diagonal() const { return max_ - min_; }
  virtual BBox get_bounding_box() const;

  //! set the mesh statistics
  void set_nx(int x) { nx_ = x; }
  void set_ny(int y) { ny_ = y; }
  void set_nz(int z) { nz_ = z; }
  void set_min(Point p) { min_ = p; }
  void set_max(Point p) { max_ = p; }

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const {}
  void get_nodes(Node::array_type &, Face::index_type) const {}
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;
  void get_edges(Edge::array_type &, Face::index_type) const {}
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  void get_faces(Face::array_type &, Cell::index_type) const {}

  //! get the parent element(s) of the given index
  int get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  int get_faces(Face::array_type &, Node::index_type) const { return 0; }
  int get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  int get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  int get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  int get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  //! similar to get_cells() with Face::index_type argument, but
  //  returns the "other" cell if it exists, not all that exist
  void get_neighbor(Cell::index_type &, Face::index_type) const {}

  //! get the center point (in object space) of an element
  void get_center(Point &result, Node::index_type idx) const;
  void get_center(Point &, Edge::index_type) const {}
  void get_center(Point &, Face::index_type) const {}
  void get_center(Point &result, Cell::index_type idx) const;


  bool locate(Node::index_type &node, const Point &p) const;
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &cell, const Point &p) const;

  void unlocate(Point &result, const Point &p) const { result =  p; };

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) {ASSERTFAIL("LatVolMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &, Face::array_type &, vector<double> &) {ASSERTFAIL("LatVolMesh::get_weights for faces isn't supported");}
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w);

  void get_point(Point &result, Node::index_type index) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const SCIRun::TypeDescription *get_type_description() const;

private:

  // each LevelMesh needs grid and level index
  GridP grid_; 
  int level_;
  IntVector idxLow_; // cache the low index


  //! the Node::index_type space extents of a LevelMesh (min=0, max=n-1)
  int nx_, ny_, nz_;

  //! the object space extents of a LevelMesh
  Point min_, max_;
  Transform transform_;
  
  // returns a LevelMesh
  static Persistent *maker() { return scinew LevelMesh(); }
};

}

namespace SCIRun {

const SCIRun::TypeDescription* get_type_description(Uintah::LevelMesh *);
const SCIRun::TypeDescription* get_type_description(Uintah::LevelMesh::Node *);
const SCIRun::TypeDescription* get_type_description(Uintah::LevelMesh::Edge *);
const SCIRun::TypeDescription* get_type_description(Uintah::LevelMesh::Face *);
const SCIRun::TypeDescription* get_type_description(Uintah::LevelMesh::Cell *);

} // namespace SCIRun

#endif // SCI_project_LevelMesh_h

