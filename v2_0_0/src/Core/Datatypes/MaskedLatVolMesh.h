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
 *  MaskedLatVolMesh.h:
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   Feb 2003
 *
 *  Copyright (C) 2002 SCI Group
 *
 */



#ifndef SCI_project_MaskedLatVolMesh_h
#define SCI_project_MaskedLatVolMesh_h 1

#include <Core/Datatypes/LatVolField.h>
#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <set>
#include <sgi_stl_warnings_on.h>

// This removes compiler warnings about unreachable statements.
// Only use BREAK after lines that will kill the program.  Otherwise
// use break.
#if defined(__sgi) && !defined(__GNUC__)
#  define BREAK 
#else
#  define BREAK break
#endif

namespace SCIRun {

using std::string;

class SCICORESHARE MaskedLatVolMesh : public LatVolMesh
{
private:

public:

  struct MaskedLatIndex;
  friend struct MaskedLatIndex;

  struct MaskedLatIndex
  {
  public:
    MaskedLatIndex() : i_(0), j_(0), k_(0), mesh_(0) {}
    MaskedLatIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, 
		   unsigned k) : i_(i), j_(j), k_(k), mesh_(m) {}
        
    unsigned i_, j_, k_;

    // Needs to be here so we can compute a sensible index.
    const MaskedLatVolMesh *mesh_;
  };

  struct CellIndex : public MaskedLatIndex
  {
    CellIndex() : MaskedLatIndex() {}
    CellIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i,j,k) {}

    operator unsigned() const { 
      ASSERT(mesh_);
      return i_ + (mesh_->ni_-1)*j_ + (mesh_->ni_-1)*(mesh_->nj_-1)*k_;
    }

    bool operator ==(const CellIndex &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }

    bool operator !=(const CellIndex &a) const
    {
      return !(*this == a);
    }


    static string type_name(int i=-1) 
    { ASSERT(i<1); return "MaskedLatVolMesh::CellIndex"; }
    friend void Pio(Piostream&, CellIndex&);
    friend const TypeDescription* get_type_description(CellIndex *);
    friend const string find_type_name(CellIndex *);
  };

  struct NodeIndex : public MaskedLatIndex
  {
    NodeIndex() : MaskedLatIndex() {}
    NodeIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i,j,k) {}

    operator unsigned() const { 
      ASSERT(mesh_);
      return i_ + mesh_->ni_*j_ + mesh_->ni_*mesh_->nj_*k_;
    }

    bool operator ==(const MaskedLatIndex &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }

    bool operator !=(const MaskedLatIndex &a) const
    {
      return !(*this == a);
    }


    static string type_name(int i=-1) 
    { ASSERT(i<1); return "MaskedLatVolMesh::NodeIndex"; }
    friend void Pio(Piostream&, NodeIndex&);
    friend const TypeDescription* get_type_description(NodeIndex *);
    friend const string find_type_name(NodeIndex *);
  };

  struct EdgeIndex : public MaskedLatIndex
  {
    EdgeIndex() : MaskedLatIndex(), dir_(0) {}
    EdgeIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k, 
	      unsigned dir)
      : MaskedLatIndex(m, i,j,k) , dir_(dir){}
    operator unsigned() const { 
      ASSERT(mesh_);
      switch (dir_)
	{
	case 0: return (i_ + (mesh_->ni_-1)*j_ + 
			(mesh_->ni_-1)*mesh_->nj_*k_); 
	case 1: return (j_ + (mesh_->nj_-1)*k_ + 
			(mesh_->nj_-1)*mesh_->nk_*i_ + 
			(mesh_->ni_-1)*mesh_->nj_*mesh_->nk_); 
	case 2: return (k_ + (mesh_->nk_-1)*i_ + 
			(mesh_->nk_-1)*mesh_->ni_*j_ +
			(mesh_->ni_-1)*mesh_->nj_*mesh_->nk_ + 
			mesh_->ni_*(mesh_->nj_-1)*mesh_->nk_); 
	default: return 0; //ASSERTFAIL("EdgeIndex dir_ off.");
	}
    }

    bool operator ==(const EdgeIndex &a) const
    {
      return (i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && 
	      mesh_ == a.mesh_ && dir_ == a.dir_);
    }

    bool operator !=(const EdgeIndex &a) const
    {
      return !(*this == a);
    }
    

    static string type_name(int i=-1) 
    { ASSERT(i<1); return "MaskedLatVolMesh::EdgeIndex"; }
    friend void Pio(Piostream&, EdgeIndex&);
    friend const TypeDescription* get_type_description(EdgeIndex *);
    friend const string find_type_name(EdgeIndex *);
    unsigned dir_;
  };

  struct FaceIndex : public MaskedLatIndex
  {
    FaceIndex() : MaskedLatIndex() {}
    FaceIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k, 
	      unsigned dir)
      : MaskedLatIndex(m, i,j,k) , dir_(dir){}
    operator unsigned() const { 
      ASSERT(mesh_);
      switch (dir_)
	{
	case 0: return (i_ + (mesh_->ni_-1)*j_ + 
			(mesh_->ni_-1)*(mesh_->nj_-1)*k_); 
	case 1: return (j_ + (mesh_->nj_-1)*k_ + 
			(mesh_->nj_-1)*(mesh_->nk_-1)*i_ + 
			(mesh_->ni_-1)*(mesh_->nj_-1)*mesh_->nk_);
	case 2: return (k_ + (mesh_->nk_-1)*i_ + 
			(mesh_->nk_-1)*(mesh_->ni_-1)*j_ +
			(mesh_->ni_-1)*(mesh_->nj_-1)*mesh_->nk_ + 
			mesh_->ni_*(mesh_->nj_-1)*(mesh_->nk_-1));
	default: return 0; //ASSERTFAIL("FaceIndex dir_ off."); 
	}
    }

    bool operator ==(const FaceIndex &a) const
    {
      return (i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && 
	      mesh_ == a.mesh_ && dir_ == a.dir_);
    }

    bool operator !=(const FaceIndex &a) const
    {
      return !(*this == a);
    }

    static string type_name(int i=-1) 
    { ASSERT(i<1); return "MaskedLatVolMesh::FaceIndex"; }
    friend void Pio(Piostream&, FaceIndex&);
    friend const TypeDescription* get_type_description(FaceIndex *);
    friend const string find_type_name(FaceIndex *);
    unsigned dir_;
  };

  struct CellSize : public MaskedLatIndex
  {
  public:
    CellSize() : MaskedLatIndex() {}
    CellSize(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned () const 
    { 
      return i_*j_*k_;
      // - (mesh_?mesh_->num_masked_cells() : 0);
    } 
  };

  struct NodeSize : public MaskedLatIndex
  {
  public:
    NodeSize() : MaskedLatIndex() {}
    NodeSize(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned () const 
    { 
      return i_*j_*k_;
      // - (mesh_?mesh_->num_masked_nodes() : 0);
    } 
  };

  struct EdgeSize : public MaskedLatIndex
  {
  public:
    EdgeSize() : MaskedLatIndex() {}
    EdgeSize(const MaskedLatVolMesh *m, unsigned i, unsigned j, 
	     unsigned k) 
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned () const 
    { 
      return (i_-1)*j_*k_ + i_*(j_-1)*k_ + i_*j_*(k_-1);
      // - (mesh_?mesh_->num_masked_edges() : 0);
    } 
  };

  struct FaceSize : public MaskedLatIndex
  {
  public:
    FaceSize() : MaskedLatIndex() {}
    FaceSize(const MaskedLatVolMesh *m, unsigned i, unsigned j, 
	     unsigned k) 
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned () const 
    { 
      return i_*(j_-1)*(k_-1) + (i_-1)*j_*(k_-1) + (i_-1)*(j_-1)*k_; 
      // - (mesh_?mesh_->num_masked_faces() : 0);
    } 
  };

  struct NodeIter : public MaskedLatIndex
  {
  public:
    NodeIter() : MaskedLatIndex() {}
    NodeIter(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i, j, k) {}

    const NodeIndex &operator *() const 
    { 
      return (const NodeIndex&)(*this); 
    }

    operator unsigned() const 
    { 
      ASSERT(mesh_);
      return i_ + mesh_->ni_*j_ + mesh_->ni_*mesh_->nj_*k_;
    }

    NodeIter &operator++()
    {
      do next(); while (!mesh_->check_valid(*this) && 
			(k_ < (mesh_->min_k_+mesh_->nk_)));
      return *this;
    }

    NodeIter &operator--()
    {
      do prev(); while (!mesh_->check_valid(*this));
      return *this;
    }
    inline void next() 
    {
      i_++;
      if (i_ >= mesh_->min_i_+mesh_->ni_) {
	i_ = mesh_->min_i_;
	j_++;
	if (j_ >=  mesh_->min_j_+mesh_->nj_) {
	  j_ = mesh_->min_j_;
	  k_++;
	}
      }
    }

    inline void prev()
    {
      if (i_ == mesh_->min_i_) {
	i_ = mesh_->min_i_ + mesh_->ni_;
	if (j_ == mesh_->min_j_) {
	  j_ = mesh_->min_j_ + mesh_->nj_;
	  ASSERTMSG(k_ != mesh_->min_k_-1, "Cant prev() from first node!");
	  k_--;
	}
	else {
	  j_--;
	}
      }
      else {
	i_--;
      }
    }
  private:
    NodeIter operator++(int)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }
    NodeIter operator--(int)
    {
      NodeIter result(*this);
      operator--();
      return result;
    }

  };

  struct EdgeIter : public EdgeIndex
  {
  public:
    EdgeIter() : EdgeIndex() {}
    EdgeIter(const MaskedLatVolMesh *m, 
	     unsigned i, unsigned j, 
	     unsigned k, unsigned dir)
      : EdgeIndex(m, i, j, k,dir) {}

    const EdgeIndex &operator *() const 
    { 
      return (const EdgeIndex&)(*this); 
    }

    bool operator ==(const EdgeIter &a) const
    {
      return (i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && 
	      mesh_ == a.mesh_ && dir_ == a.dir_);
    }

    bool operator !=(const EdgeIter &a) const
    {
      return !(*this == a);
    }


    EdgeIter &operator++()
    {
      do next(); while (!mesh_->check_valid(*this) && dir_ < 3);
      return *this;
    }

    EdgeIter &operator--()
    {
      do prev(); while (!mesh_->check_valid(*this));
      return *this;
    }

    inline void next() 
    {
      switch (dir_)
      {
      case 0:
	i_++;
	if (i_ >= mesh_->min_i_+mesh_->ni_-1) {
	  i_ = mesh_->min_i_;
	  j_++;
	  if (j_ >=  mesh_->min_j_+mesh_->nj_) {
	    j_ = mesh_->min_j_;
	    k_++;	 
	    if (k_ >= mesh_->min_k_+mesh_->nk_) {
	      dir_++;
	      i_ = 0;
	      j_ = 0;
	      k_ = 0;
	    }
	  }
	}
	break;
      case 1:
	j_++;
	if (j_ >= mesh_->min_j_+mesh_->nj_-1) {
	  j_ = mesh_->min_j_;
	  k_++;
	  if (k_ >=  mesh_->min_k_+mesh_->nk_) {
	    k_ = mesh_->min_k_;
	    i_++;	 
	    if (i_ >= mesh_->min_i_+mesh_->ni_) {
	      dir_++;
	      i_ = 0;
	      j_ = 0;
	      k_ = 0;
	    }
	  }
	}
	break;

      case 2:
	k_++;
	if (k_ >= mesh_->min_k_+mesh_->nk_-1) {
	  k_ = mesh_->min_k_;
	  i_++;
	  if (i_ >=  mesh_->min_i_+mesh_->ni_) {
	    i_ = mesh_->min_i_;
	    j_++;	 
	    if (j_ >= mesh_->min_j_+mesh_->nj_) {
	      dir_++;
	      i_ = 0;
	      j_ = 0;
	      k_ = 0;
	    }
	  }
	}
	break;
      default:
      case 3:
	ASSERTFAIL("Iterating beyond MaskedLatVolMesh edge boundaries.");
	BREAK;
      }
    }

    inline void prev()
    {
      switch(dir_)
	{
	case 2:
	  if (k_ == mesh_->min_k_) {
	    k_ = mesh_->min_k_ + mesh_->nk_-1;
	    if (i_ == mesh_->min_i_) {
	      i_ = mesh_->min_i_ + mesh_->ni_;
	      if (j_ == mesh_->min_j_) {
		i_ = mesh_->min_i_ + mesh_->ni_;
		j_ = mesh_->min_j_ + mesh_->nj_-1;
		k_ = mesh_->min_k_ + mesh_->nk_;
		dir_--;
	      }
	      else {
		j_--;
	      }
	    }
	    else {
	      i_--;
	    }
	  }
	  else {
	    k_--;
	  }
	  break;

	case 1:
	  if (j_ == mesh_->min_j_) {
	    j_ = mesh_->min_j_ + mesh_->nj_-1;
	    if (k_ == mesh_->min_k_) {
	      k_ = mesh_->min_k_ + mesh_->nk_;
	      if (i_ == mesh_->min_i_) {
		i_ = mesh_->min_i_ + mesh_->ni_-1;
		j_ = mesh_->min_j_ + mesh_->nj_;
		k_ = mesh_->min_k_ + mesh_->nk_;
		dir_--;
	      }
	      else {
		i_--;
	      }
	    }
	    else {
	      k_--;
	    }
	  }
	  else {
	    j_--;
	  }
	  break;

	case 0:
	  if (i_ == mesh_->min_i_) {
	    i_ = mesh_->min_i_ + mesh_->ni_-1;
	    if (j_ == mesh_->min_j_) {
	      j_ = mesh_->min_j_ + mesh_->nj_;
	      if (k_ == mesh_->min_k_) {
		ASSERTFAIL("Iterating b4 MaskedLatVolMesh edge boundaries.");
	      }
	      else {
		k_--;
	      }
	    }
	    else {
	      j_--;
	    }
	  }
	  else {
	    i_--;
	  }
	  break;
	default:
	  ASSERTFAIL("Iterating beyond MaskedLatVolMesh edge boundaries.");
	  BREAK;
	}
    }
  private:    
    EdgeIter operator++(int)
    {
      EdgeIter result(*this);
      operator++();
      return result;
    }
    EdgeIter operator--(int)
    {
      EdgeIter result(*this);
      operator--();
      return result;
    }

  };


  struct FaceIter : public FaceIndex
  {
  public:
    FaceIter() : FaceIndex() {}
    FaceIter(const MaskedLatVolMesh *m, 
	     unsigned i, unsigned j, unsigned k, unsigned dir)
      : FaceIndex(m, i, j, k, dir){}

    const FaceIndex &operator *() const 
    { 
      return (const FaceIndex&)(*this); 
    }

    FaceIter &operator++()
    {
      do next(); while (!mesh_->check_valid(*this) && dir_ < 3);
      return *this;
    }

    FaceIter &operator--()
    {
      do prev(); while (!mesh_->check_valid(*this));
      return *this;
    }

    inline void next() 
    {
      switch (dir_)
      {
      case 0:
	i_++;
	if (i_ >= mesh_->min_i_+mesh_->ni_-1) {
	  i_ = mesh_->min_i_;
	  j_++;
	  if (j_ >=  mesh_->min_j_+mesh_->nj_-1) {
	    j_ = mesh_->min_j_;
	    k_++;	 
	    if (k_ >= mesh_->min_k_+mesh_->nk_) {
	      dir_++;
	      i_ = 0;
	      j_ = 0;
	      k_ = 0;
	    }
	  }
	}
	break;
      case 1:
	j_++;
	if (j_ >= mesh_->min_j_+mesh_->nj_-1) {
	  j_ = mesh_->min_j_;
	  k_++;
	  if (k_ >=  mesh_->min_k_+mesh_->nk_-1) {
	    k_ = mesh_->min_k_;
	    i_++;	 
	    if (i_ >= mesh_->min_i_+mesh_->ni_) {
	      dir_++;
	      i_ = 0;
	      j_ = 0;
	      k_ = 0;
	    }
	  }
	}
	break;

      case 2:
	k_++;
	if (k_ >= mesh_->min_k_+mesh_->nk_-1) {
	  k_ = mesh_->min_k_;
	  i_++;
	  if (i_ >=  mesh_->min_i_+mesh_->ni_-1) {
	    i_ = mesh_->min_i_;
	    j_++;	 
	    if (j_ >= mesh_->min_j_+mesh_->nj_) {
	      dir_++;
	      i_ = 0;
	      j_ = 0;
	      k_ = 0;
	    }
	  }
	}
	break;
      default:
      case 3:
	ASSERTFAIL("Iterating beyond MaskedLatVolMesh edge boundaries.");
	BREAK;
      }
    }

    inline void prev()
    {
      switch(dir_)
	{
	case 2:
	  if (k_ == mesh_->min_k_) {
	    k_ = mesh_->min_k_ + mesh_->nk_-1;
	    if (i_ == mesh_->min_i_) {
	      i_ = mesh_->min_i_ + mesh_->ni_-1;
	      if (j_ == mesh_->min_j_) {
		i_ = mesh_->min_i_ + mesh_->ni_;
		j_ = mesh_->min_j_ + mesh_->nj_-1;
		k_ = mesh_->min_k_ + mesh_->nk_-1;
		dir_--;
	      }
	      else {
		j_--;
	      }
	    }
	    else {
	      i_--;
	    }
	  }
	  else {
	    k_--;
	  }
	  break;

	case 1:
	  if (j_ == mesh_->min_j_) {
	    j_ = mesh_->min_j_ + mesh_->nj_-1;
	    if (k_ == mesh_->min_k_) {
	      k_ = mesh_->min_k_ + mesh_->nk_-1;
	      if (i_ == mesh_->min_i_) {
		i_ = mesh_->min_i_ + mesh_->ni_-1;
		j_ = mesh_->min_j_ + mesh_->nj_-1;
		k_ = mesh_->min_k_ + mesh_->nk_;
		dir_--;
	      }
	      else {
		i_--;
	      }
	    }
	    else {
	      k_--;
	    }
	  }
	  else {
	    j_--;
	  }
	  break;

	case 0:
	  if (i_ == mesh_->min_i_) {
	    i_ = mesh_->min_i_ + mesh_->ni_-1;
	    if (j_ == mesh_->min_j_) {
	      j_ = mesh_->min_j_ + mesh_->nj_-1;
	      if (k_ == mesh_->min_k_) {
		ASSERTFAIL("Iterating b4 MaskedLatVolMesh face boundaries.");
	      }
	      else {
		k_--;
	      }
	    }
	    else {
	      j_--;
	    }
	  }
	  else {
	    i_--;
	  }
	  break;
	default:
	  ASSERTFAIL("Iterating beyond MaskedLatVolMesh face boundaries.");
	  BREAK;
	}
    }

  private:
    FaceIter operator++(int)
    {
      FaceIter result(*this);
      operator++();
      return result;
    }
    FaceIter operator--(int)
    {
      FaceIter result(*this);
      operator--();
      return result;
    }

  };




  struct CellIter : public MaskedLatIndex
  {
  public:
    CellIter() : MaskedLatIndex() {}
    CellIter(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i, j, k) {}

    const CellIndex &operator *() const 
    { 
      return (const CellIndex&)(*this); 
    }

    operator unsigned() const 
    { 
      ASSERT(mesh_);
      return i_ + (mesh_->ni_-1)*j_ + (mesh_->ni_-1)*(mesh_->nj_-1)*k_;
    }

    CellIter &operator++()
    {
      do next(); while (!mesh_->check_valid(i_,j_,k_) && 
			k_ < mesh_->min_k_ + mesh_->nk_ - 1);
      return *this;
    }

    CellIter &operator--()
    {
      do prev(); while (!mesh_->check_valid(i_,j_,k_));
      return *this;
    }

    inline void next() 
    {
      i_++;
      if (i_ >= mesh_->min_i_+mesh_->ni_-1)	{
	i_ = mesh_->min_i_;
	j_++;
	if (j_ >=  mesh_->min_j_+mesh_->nj_-1) {
	  j_ = mesh_->min_j_;
	  k_++;
	}
      }
    }

    inline void prev()
    {
      if (i_ == mesh_->min_i_) {
	i_ = mesh_->min_i_ + mesh_->ni_-1;
	if (j_ == mesh_->min_j_) {
	  j_ = mesh_->min_j_ + mesh_->nj_-1;
	  ASSERTMSG(k_ != mesh_->min_k_, "Cant prev() from first cell!");
	  k_--;
	}
	else {
	  j_--;
	}
      }
      else {
	i_--;
      }
    }
  private:      
    CellIter operator++(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }
    CellIter operator--(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }

  };



  struct Node {
    typedef NodeIndex		index_type;
    typedef NodeIter            iterator;
    typedef NodeSize            size_type;
    typedef vector<index_type>  array_type;
  };			
  			
  struct Edge {		
    typedef EdgeIndex		index_type;
    typedef EdgeIter		iterator;
    typedef EdgeSize		size_type;
    typedef vector<index_type>  array_type;
  };			
			
  struct Face {		
    typedef FaceIndex           index_type;
    typedef FaceIter            iterator;
    typedef FaceSize            size_type;
    typedef vector<index_type>  array_type;
  };			
			
  struct Cell {		
    typedef CellIndex           index_type;
    typedef CellIter            iterator;
    typedef CellSize            size_type;
    typedef vector<index_type>  array_type;
  };

  typedef Cell Elem;

  friend class NodeIter;
  friend class CellIter;
  friend class EdgeIter;
  friend class FaceIter;

  friend class NodeIndex;
  friend class CellIndex;
  friend class EdgeIndex;
  friend class FaceIndex;

  MaskedLatVolMesh();
  MaskedLatVolMesh(unsigned int x, unsigned int y, unsigned int z,
		   const Point &min, const Point &max);
  MaskedLatVolMesh(const MaskedLatVolMesh &copy);
  virtual MaskedLatVolMesh *clone() { return new MaskedLatVolMesh(*this); }
  virtual ~MaskedLatVolMesh() {}

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;

  //! Methods specific to MaskedLatVolMesh
  void mask_node(Node::index_type);
  void mask_edge(Edge::index_type);
  void mask_face(Face::index_type);
  void mask_cell(Cell::index_type);
  void unmask_node(Node::index_type);
  void unmask_edge(Edge::index_type);
  void unmask_face(Face::index_type);
  void unmask_cell(Cell::index_type);
  //! Special Method to Reset Mesh
  void unmask_everything();  
  

  unsigned num_masked_nodes() const;
  unsigned num_masked_edges() const;
  unsigned num_masked_faces() const;
  unsigned num_masked_cells() const;

  void begin(Node::iterator &) const;
  void begin(Edge::iterator &) const;
  void begin(Face::iterator &) const;
  void begin(Cell::iterator &) const;

  void end(Node::iterator &) const;
  void end(Edge::iterator &) const;
  void end(Face::iterator &) const;
  void end(Cell::iterator &) const;

  void size(Node::size_type &) const;
  void size(Edge::size_type &) const;
  void size(Face::size_type &) const;
  void size(Cell::size_type &) const;

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const;
  void get_nodes(Node::array_type &, Cell::index_type) const;
  void get_edges(Edge::array_type &, Face::index_type) const; 
  void get_edges(Edge::array_type &, Cell::index_type) const;
  void get_faces(Face::array_type &, Cell::index_type) const;

  // returns 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
							  Cell::index_type idx) const;
  // return 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Node::index_type> > &nbrs, 
							  Node::index_type idx) const;
  
  // return 8 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
							  Node::index_type idx) const;


  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, const Edge::index_type &) const;
  void get_center(Point &, const Face::index_type &) const;
  void get_center(Point &, const Cell::index_type &) const;

  double get_size(Node::index_type idx) const;
  double get_size(Edge::index_type idx) const;
  double get_size(Face::index_type idx) const;
  double get_size(Cell::index_type idx) const;
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &);

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w);

  void get_point(Point &point, const Node::index_type &index) const
  { get_center(point, index); }

  void get_random_point(Point &p, const Elem::index_type &ei, int seed=0) const
  { ASSERTFAIL("not implemented") }

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(Cell::index_type ci, Vector& g0, Vector& g1,
			    Vector& g2, Vector& g3, Vector& g4,
			    Vector& g5, Vector& g6, Vector& g7)
  { ASSERTFAIL("not implemented") }

  void get_normal(Vector &/*normal*/, Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }


  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

  unsigned int	get_sequential_node_index(const Node::index_type idx);

private:
  unsigned int	synchronized_;
  map<Node::index_type, unsigned>	nodes_;
  set<unsigned>	masked_cells_;
  unsigned	masked_nodes_count_;
  unsigned	masked_edges_count_;
  unsigned	masked_faces_count_;

  bool		update_count(Cell::index_type, bool masking);
  unsigned	num_missing_faces(Cell::index_type);
  bool		check_valid(Node::index_type idx) const;
  bool		check_valid(Edge::index_type idx) const;
  bool		check_valid(Face::index_type idx) const;
  bool		check_valid(Cell::index_type idx) const;
  bool		check_valid(Node::iterator idx) const;
  bool		check_valid(Edge::iterator idx) const;
  bool		check_valid(Face::iterator idx) const;    
  bool		check_valid(Cell::iterator idx) const; 
#if 0
  inline bool	check_valid(unsigned i, unsigned j, unsigned k) const
  { 
    return (masked_cells_.find(unsigned(Cell::index_type(this,i,j,k))) == masked_cells_.end()); 
  }
#endif
  inline bool	check_valid(int i, int j, int k) const
  { 
    if ((i >= int(min_i_)) && (i < (int(min_i_ + ni_) - 1)) &&
	(j >= int(min_j_)) && (j < (int(min_j_ + nj_) - 1)) &&
	(k >= int(min_k_)) && (k < (int(min_k_ + nk_) - 1)) &&
	(masked_cells_.find(unsigned(Cell::index_type(this,i,j,k))) == masked_cells_.end()))
      {
	return true;
      }
    return false;
  }
  
  
  // returns a MaskedLatVolMesh
  static Persistent *maker() { return new MaskedLatVolMesh(); }
};

typedef LockingHandle<MaskedLatVolMesh> MaskedLatVolMeshHandle;

const TypeDescription* get_type_description(MaskedLatVolMesh *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Node *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Edge *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Face *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Cell *);

const TypeDescription* get_type_description(MaskedLatVolMesh::CellIndex *);

} // namespace SCIRun

#endif // SCI_project_MaskedLatVolMesh_h
