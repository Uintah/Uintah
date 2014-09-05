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

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/MeshBase.h>
#include <Core/share/share.h>
#include <string>
#include <iostream>

namespace SCIRun {

using std::string;
using std::cerr;
using std::endl;

class SCICORESHARE LatVolMesh : public MeshBase
{
public:

  struct UnfinishedIndex
  {
  public:
    UnfinishedIndex() : i_(0) {}
    UnfinishedIndex(unsigned i) : i_(i) {}

    operator unsigned() const { return i_; }

    unsigned i_;
  };

  struct EdgeIndex : public UnfinishedIndex
  {
    EdgeIndex() : UnfinishedIndex() {}
    EdgeIndex(unsigned i) : UnfinishedIndex(i) {}
  };

  struct FaceIndex : public UnfinishedIndex
  {
    FaceIndex() : UnfinishedIndex() {}
    FaceIndex(unsigned i) : UnfinishedIndex(i) {}
  };

  struct UnfinishedIter : public UnfinishedIndex
  {
    UnfinishedIter(const LatVolMesh *m, unsigned i) 
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
    
    const LatVolMesh *mesh_;
  };

  struct EdgeIter : public UnfinishedIter
  {
    EdgeIter(const LatVolMesh *m, unsigned i) 
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
    FaceIter(const LatVolMesh *m, unsigned i) 
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

  struct LatIndex
  {
  public:
    LatIndex() : i_(0), j_(0), k_(0) {}
    LatIndex(unsigned i, unsigned j, unsigned k) : i_(i), j_(j), k_(k) {}
    
    operator unsigned() const { return i_*j_*k_; }

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
  
  struct LatIter : public LatIndex
  {
    LatIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : LatIndex(i, j, k), mesh_(m) {}
    
    const LatIndex &operator *() { return *this; }
    
    bool operator ==(const LatIter &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }
    
    bool operator !=(const LatIter &a) const
    {
      return !(*this == a);
    }
    
    const LatVolMesh *mesh_;
  };
  
  
  struct NodeIter : public LatIter
  {
    NodeIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : LatIter(m, i, j, k) {}
    
    const NodeIndex &operator *() const { return (const NodeIndex&)(*this); }

    NodeIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->nx_)	{
	i_ = 0;
	j_++;
	if (j_ >= mesh_->ny_) {
	  j_ = 0;
	  k_++;
	}
      }
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
  
  
  struct CellIter : public LatIter
  {
    CellIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : LatIter(m, i, j, k) {}
    
    const CellIndex &operator *() const { return (const CellIndex&)(*this); }

    CellIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->nx_-1) {
	i_ = 0;
	j_++;
	if (j_ >= mesh_->ny_-1) {
	  j_ = 0;
	  k_++;
	}
      }
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
  
  typedef LatIndex index_type;
  
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

  LatVolMesh()
    : nx_(1),ny_(1),nz_(1),min_(Point(0,0,0)),max_(Point(1,1,1)) {}
  LatVolMesh(unsigned x, unsigned y, unsigned z, 
	     const Point &min, const Point &max) 
    : nx_(x),ny_(y),nz_(z),min_(min),max_(max) {}
  LatVolMesh(const LatVolMesh &copy)
    : nx_(copy.get_nx()),ny_(copy.get_ny()),nz_(copy.get_nz()),
      min_(copy.get_min()),max_(copy.get_max()) {}
  virtual LatVolMesh *clone() { return new LatVolMesh(*this); }
  virtual ~LatVolMesh() {}

  node_iterator  node_begin() const { return node_iterator(this, 0, 0, 0); }
  node_iterator  node_end() const { return node_iterator(this, 0, 0, nz_); }
  node_size_type nodes_size() const { return node_size_type(nx_,ny_,nz_); }

  edge_iterator  edge_begin() const { return edge_iterator(this,0); }
  edge_iterator  edge_end() const { return edge_iterator(this,0); }
  edge_size_type edges_size() const { return edge_size_type(0); }

  face_iterator  face_begin() const { return face_iterator(this,0); }
  face_iterator  face_end() const { return face_iterator(this,0); }
  face_size_type faces_size() const { return face_size_type(0); }

  cell_iterator  cell_begin() const { return cell_iterator(this, 0, 0, 0); }
  cell_iterator  cell_end() const { return cell_iterator(this, 0, 0, nz_-1); }
  cell_size_type cells_size() const { return cell_size_type(nx_-1,
							    ny_-1,nz_-1); }

  //! get the mesh statistics
  unsigned get_nx() const { return nx_; }
  unsigned get_ny() const { return ny_; }
  unsigned get_nz() const { return nz_; }
  Point get_min() const { return min_; }
  Point get_max() const { return max_; }
  Vector diagonal() const { return max_-min_; }
  virtual BBox get_bounding_box() const;

  //! set the mesh statistics
  void set_nx(unsigned x) { nx_ = x; }
  void set_ny(unsigned y) { ny_ = y; }
  void set_nz(unsigned z) { nz_ = z; }
  void set_min(Point p) { min_ = p; }
  void set_max(Point p) { max_ = p; }


  //! get the child elements of the given index
  void get_nodes(node_array &, edge_index) const {}
  void get_nodes(node_array &, face_index) const {}
  void get_nodes(node_array &, cell_index) const;
  void get_edges(edge_array &, face_index) const {}
  void get_edges(edge_array &, cell_index) const {}
  void get_faces(face_array &, cell_index) const {}

  //! get the parent element(s) of the given index
  unsigned get_edges(edge_array &, node_index) const { return 0; }
  unsigned get_faces(face_array &, node_index) const { return 0; }
  unsigned get_faces(face_array &, edge_index) const { return 0; }
  unsigned get_cells(cell_array &, node_index) const { return 0; }
  unsigned get_cells(cell_array &, edge_index) const { return 0; }
  unsigned get_cells(cell_array &, face_index) const { return 0; }

  //! return all cell_indecies that overlap the BBox in arr.
  void get_cells(cell_array &arr, const BBox &box) const;

  //! similar to get_cells() with face_index argument, but
  //  returns the "other" cell if it exists, not all that exist
  bool get_neighbor(cell_index & /*neighbor*/, cell_index /*from*/, 
		    face_index /*idx*/) const {
    ASSERTFAIL("LatVolMesh::get_neighbor not implemented.");
  }
  //! get the center point (in object space) of an element
  void get_center(Point &, node_index) const;
  void get_center(Point &, edge_index) const {}
  void get_center(Point &, face_index) const {}
  void get_center(Point &, cell_index) const;

  bool locate(node_index &, const Point &) const;
  bool locate(edge_index &, const Point &) const { return false; }
  bool locate(face_index &, const Point &) const { return false; }
  bool locate(cell_index &, const Point &) const;

  void get_point(Point &p, node_index i) const
  { get_center(p, i); }
    

  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

private:

  //! the node_index space extents of a LatVolMesh (min=0, max=n-1)
  unsigned nx_, ny_, nz_;

  //! the object space extents of a LatVolMesh
  Point min_, max_;

  // returns a LatVolMesh
  static Persistent *maker() { return new LatVolMesh(); }
};

typedef LockingHandle<LatVolMesh> LatVolMeshHandle;



} // namespace SCIRun

#endif // SCI_project_LatVolMesh_h
