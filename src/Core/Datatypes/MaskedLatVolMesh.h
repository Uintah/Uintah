/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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

#include <Core/Datatypes/LatVolMesh.h>
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

template <class Basis>
class MaskedLatVolMesh : public LatVolMesh<Basis>
{
public:
  typedef Basis                                   basis_type;
  typedef LockingHandle<MaskedLatVolMesh<Basis> > handle_type;

  struct MaskedLatIndex;
  friend struct MaskedLatIndex;

  struct MaskedLatIndex
  {
  public:
    MaskedLatIndex() : i_(0), j_(0), k_(0), mesh_(0) {}
    MaskedLatIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, 
		   unsigned k) : i_(i), j_(j), k_(k), mesh_(m) {}
        
    operator unsigned() const { 
      ASSERT(mesh_);
      return i_ + ni()*j_ + ni()*nj()*k_;
    }
    
    // Make sure mesh_ is valid before calling these convience accessors
    unsigned ni() const { ASSERT(mesh_); return mesh_->get_ni(); }
    unsigned nj() const { ASSERT(mesh_); return mesh_->get_nj(); }
    unsigned nk() const { ASSERT(mesh_); return mesh_->get_nk(); }

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
      return i_ + (ni()-1)*j_ + (ni()-1)*(nj()-1)*k_;
    }

    friend void Pio<Basis>(Piostream&, CellIndex&);
    //friend const TypeDescription* get_type_description(CellIndex *);
    friend const string find_type_name<Basis>(CellIndex *);
  };

  struct NodeIndex : public MaskedLatIndex
  {
    NodeIndex() : MaskedLatIndex() {}
    NodeIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i,j,k) {}

    static string type_name(int i=-1) 
    { 
      ASSERT(i<1); 
      return MaskedLatVolMesh::type_name(-1) + "::NodeIndex"; 
    }
    friend void Pio<Basis>(Piostream&, NodeIndex&);
    //friend const TypeDescription* get_type_description(NodeIndex *);
    friend const string find_type_name<Basis>(NodeIndex *);
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
    { 
      ASSERT(i<1); 
      return MaskedLatVolMesh::type_name(-1) + "::EdgeIndex"; 
    }
    friend void Pio<Basis>(Piostream&, EdgeIndex&);
    //friend const TypeDescription* get_type_description(EdgeIndex *);
    //friend const string find_type_name<Basis>(EdgeIndex *);
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
    { 
      ASSERT(i<1); 
      return MaskedLatVolMesh::type_name(-1) + "::FaceIndex"; 
    }
    friend void Pio<Basis>(Piostream&, FaceIndex&);
    //friend const TypeDescription* get_type_description(FaceIndex *);
    //friend const string find_type_name<Basis>(FaceIndex *);
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
  void mask_node(typename Node::index_type);
  void mask_edge(typename Edge::index_type);
  void mask_face(typename Face::index_type);
  void mask_cell(typename Cell::index_type);
  void unmask_node(typename Node::index_type);
  void unmask_edge(typename Edge::index_type);
  void unmask_face(typename Face::index_type);
  void unmask_cell(typename Cell::index_type);
  //! Special Method to Reset Mesh
  void unmask_everything();  
  

  unsigned num_masked_nodes() const;
  unsigned num_masked_edges() const;
  unsigned num_masked_faces() const;
  unsigned num_masked_cells() const;

  void begin(typename Node::iterator &) const;
  void begin(typename Edge::iterator &) const;
  void begin(typename Face::iterator &) const;
  void begin(typename Cell::iterator &) const;

  void end(typename Node::iterator &) const;
  void end(typename Edge::iterator &) const;
  void end(typename Face::iterator &) const;
  void end(typename Cell::iterator &) const;

  void size(typename Node::size_type &) const;
  void size(typename Edge::size_type &) const;
  void size(typename Face::size_type &) const;
  void size(typename Cell::size_type &) const;

  void to_index(typename Node::index_type &index, unsigned int i);
  void to_index(typename Edge::index_type &index, unsigned int i);
  void to_index(typename Face::index_type &index, unsigned int i);
  void to_index(typename Cell::index_type &index, unsigned int i);

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, typename Face::index_type) const;
  void get_nodes(typename Node::array_type &, typename Cell::index_type) const;
  void get_edges(typename Edge::array_type &, typename Face::index_type) const; 
  void get_edges(typename Edge::array_type &, typename Cell::index_type) const;
  void get_faces(typename Face::array_type &, typename Cell::index_type) const;

  // returns 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,typename Cell::index_type> > &nbrs, 
							  typename Cell::index_type idx) const;
  // return 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,typename Node::index_type> > &nbrs, 
							  typename Node::index_type idx) const;
  
  // return 8 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,typename Cell::index_type> > &nbrs, 
							  typename Node::index_type idx) const;


  //! get the center point (in object space) of an element
  void get_center(Point &, const typename Node::index_type &) const;
  void get_center(Point &, const typename Edge::index_type &) const;
  void get_center(Point &, const typename Face::index_type &) const;
  void get_center(Point &, const typename Cell::index_type &) const;

  double get_size(typename Node::index_type idx) const;
  double get_size(typename Edge::index_type idx) const;
  double get_size(typename Face::index_type idx) const;
  double get_size(typename Cell::index_type idx) const;
  double get_length(typename Edge::index_type idx) const { return get_size(idx); };
  double get_area(typename Face::index_type idx) const { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const { return get_size(idx); };

  bool locate(typename Node::index_type &, const Point &);
  bool locate(typename Edge::index_type &, const Point &) const { return false; }
  bool locate(typename Face::index_type &, const Point &) const { return false; }
  bool locate(typename Cell::index_type &, const Point &);

  void get_point(Point &point, const typename Node::index_type &index) const
  { get_center(point, index); }

  void get_random_point(Point &/*p*/, const typename Elem::index_type &/*ei*/, int /*seed=0*/) const
  { ASSERTFAIL("not implemented") }

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(typename Cell::index_type /*ci*/, Vector& /*g0*/, Vector& /*g1*/,
			    Vector& /*g2*/, Vector& /*g3*/, Vector& /*g4*/,
			    Vector& /*g5*/, Vector& /*g6*/, Vector& /*g7*/)
  { ASSERTFAIL("not implemented") }

  void get_normal(Vector &/*normal*/, typename Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }


  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);

  virtual const TypeDescription *get_type_description() const;
  static const TypeDescription* cell_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* node_type_description();
  static const TypeDescription* cell_index_type_description();
  static const TypeDescription* node_index_type_description();

  unsigned int	get_sequential_node_index(const typename Node::index_type idx);

  // returns a MaskedLatVolMesh
  static Persistent *maker() { return new MaskedLatVolMesh<Basis>(); }

private:
  unsigned int	synchronized_;
  map<typename Node::index_type, unsigned>	nodes_;
  Mutex					node_lock_;
  set<unsigned>	masked_cells_;
  unsigned	masked_nodes_count_;
  unsigned	masked_edges_count_;
  unsigned	masked_faces_count_;

  bool		update_count(typename Cell::index_type, bool masking);
  unsigned	num_missing_faces(typename Cell::index_type);
  bool		check_valid(typename Node::index_type idx) const;
  bool		check_valid(typename Edge::index_type idx) const;
  bool		check_valid(typename Face::index_type idx) const;
  bool		check_valid(typename Cell::index_type idx) const;
  bool		check_valid(typename Node::iterator idx) const;
  bool		check_valid(typename Edge::iterator idx) const;
  bool		check_valid(typename Face::iterator idx) const;    
  bool		check_valid(typename Cell::iterator idx) const; 
#if 0
  inline bool	check_valid(unsigned i, unsigned j, unsigned k) const
  { 
    return (masked_cells_.find(unsigned(typename Cell::index_type(this,i,j,k))) == masked_cells_.end()); 
  }
#endif
  inline bool	check_valid(int i, int j, int k) const
  { 
    if ((i >= int(min_i_)) && (i < (int(min_i_ + ni_) - 1)) &&
	(j >= int(min_j_)) && (j < (int(min_j_ + nj_) - 1)) &&
	(k >= int(min_k_)) && (k < (int(min_k_ + nk_) - 1)) &&
	(masked_cells_.find(unsigned(typename Cell::index_type(this,i,j,k))) == masked_cells_.end()))
      {
	return true;
      }
    return false;
  }
};

template <class Basis>
PersistentTypeID 
MaskedLatVolMesh<Basis>::type_id(type_name(-1), 
				 LatVolMesh<Basis>::type_name(-1), 
				 maker);

template <class Basis>
MaskedLatVolMesh<Basis>::MaskedLatVolMesh():
  LatVolMesh<Basis>(),
  synchronized_(0),
  nodes_(),
  node_lock_("MaskedLatVolMesh node_lock_"),
  masked_cells_(), 
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)

{}

template <class Basis>
MaskedLatVolMesh<Basis>::MaskedLatVolMesh(unsigned int x,
				   unsigned int y,
				   unsigned int z,
				   const Point &min,
				   const Point &max) :
  LatVolMesh<Basis>(x, y, z, min, max),
  synchronized_(0),
  nodes_(),
  node_lock_("MaskedLatVolMesh node_lock_"),
  masked_cells_(), 
  masked_nodes_count_(0),
  masked_edges_count_(0),
  masked_faces_count_(0)
{}

template <class Basis>
MaskedLatVolMesh<Basis>::MaskedLatVolMesh(const MaskedLatVolMesh<Basis> &copy):
  LatVolMesh<Basis>(copy),
  synchronized_(copy.synchronized_),
  nodes_(copy.nodes_),
  node_lock_("MaskedLatVolMesh node_lock_"),
  masked_cells_(copy.masked_cells_), 
  masked_nodes_count_(copy.masked_nodes_count_),
  masked_edges_count_(copy.masked_edges_count_),
  masked_faces_count_(copy.masked_edges_count_)
{
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Node::iterator &itr) const
{
  itr = typename Node::iterator(this, min_i_, min_j_, min_k_);
  if (!check_valid(itr)) ++itr;
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Node::iterator &itr) const
{
  itr = typename Node::iterator(this, min_i_, min_j_, min_k_ + nk_);
  //  if (!check_valid(itr)) { --itr; itr.next(); }  
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Node::size_type &s) const
{
  s = typename Node::size_type(this,ni_,nj_,nk_);
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Node::index_type &idx,
				  unsigned int a)
{
  const unsigned int i = a % ni_;
  const unsigned int jk = a / ni_;
  const unsigned int j = jk % nj_;
  const unsigned int k = jk / nj_;
  idx = typename Node::index_type(this, i, j, k);
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(this,  min_i_, min_j_, min_k_);
  if (!check_valid(itr)) ++itr;
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(this, min_i_, min_j_, min_k_ + nk_-1);
  //  if (!check_valid(itr)) { --itr; itr.next(); }
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Cell::size_type &s) const
{
  s = typename Cell::size_type(this,ni_-1, nj_-1,nk_-1);
}



template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Cell::index_type &idx,
				  unsigned int a)
{
  const unsigned int i = a % (ni_-1);
  const unsigned int jk = a / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  idx = typename Cell::index_type(this, i, j, k);
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Edge::iterator &itr) const
{
  itr = typename Edge::iterator(this,min_i_,min_j_,min_k_,0);
  if (!check_valid(itr)) ++itr;
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Edge::iterator &itr) const
{
  itr = typename Edge::iterator(this, min_i_, min_j_, min_k_,3);
  //  if (!check_valid(itr)) { --itr; itr.next(); }
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Edge::size_type &s) const
{
  s = typename Edge::size_type(this,ni_,nj_,nk_);
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Edge::index_type &/*idx*/,
				  unsigned int /*a*/)
{
  // TODO: Implement inverse of unsigned() function in EdgeIndex.
  ASSERTFAIL("NOT IMPLEMENTED YET!");
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::begin(typename MaskedLatVolMesh<Basis>::Face::iterator &itr) const
{
  itr = typename Face::iterator(this,min_i_,min_j_,min_k_,0);
  if (!check_valid(itr)) ++itr;
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::end(typename MaskedLatVolMesh<Basis>::Face::iterator &itr) const
{
  itr = typename Face::iterator(this, min_i_, min_j_, min_k_,3);
  //  if (!check_valid(itr)) { --itr; itr.next(); }
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::size(typename MaskedLatVolMesh<Basis>::Face::size_type &s) const
{
  s = typename Face::size_type(this,ni_,nj_,nk_);
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::to_index(typename MaskedLatVolMesh<Basis>::Face::index_type &/*idx*/,
				  unsigned int /*a*/)
{
  // TODO: Implement inverse of unsigned() function in FaceIndex.
  ASSERTFAIL("NOT IMPLEMENTED YET!");
}

//! get the child elements of the given index
template <class Basis>
void
MaskedLatVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Edge::index_type e) const
{
  array.resize(2);
  array[0] = typename Node::index_type(this,e.i_,e.j_,e.k_);
  array[1] = typename Node::index_type(this,
			      e.i_ + (e.dir_ == 0 ? 1:0),
			      e.j_ + (e.dir_ == 1 ? 1:0),
			      e.k_ + (e.dir_ == 2 ? 1:0));
}

		           
template <class Basis>
void 
MaskedLatVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Face::index_type f) const
{
  array.resize(4);
  array[0] = typename Node::index_type(this,f.i_,f.j_,f.k_);
  array[1] = typename Node::index_type(this,
			      f.i_ + (f.dir_ == 0 ? 1:0),
			      f.j_ + (f.dir_ == 1 ? 1:0),
			      f.k_ + (f.dir_ == 2 ? 1:0));
  array[2] = typename Node::index_type(this,
			      f.i_ + ((f.dir_ == 0 || f.dir_ == 2) ? 1:0),
			      f.j_ + ((f.dir_ == 0 || f.dir_ == 1) ? 1:0),
			      f.k_ + ((f.dir_ == 1 || f.dir_ == 2) ? 1:0));
  array[3] = typename Node::index_type(this,
			      f.i_ + (f.dir_ == 2 ? 1:0),
			      f.j_ + (f.dir_ == 0 ? 1:0),
			      f.k_ + (f.dir_ == 1 ? 1:0));

}

template <class Basis>
void 
MaskedLatVolMesh<Basis>::get_nodes(typename Node::array_type &array, typename Cell::index_type idx) const
{
  array.resize(8);
  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;   array[0].k_ = idx.k_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;   array[1].k_ = idx.k_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1; array[2].k_ = idx.k_;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1; array[3].k_ = idx.k_;
  array[4].i_ = idx.i_;   array[4].j_ = idx.j_;   array[4].k_ = idx.k_+1;
  array[5].i_ = idx.i_+1; array[5].j_ = idx.j_;   array[5].k_ = idx.k_+1;
  array[6].i_ = idx.i_+1; array[6].j_ = idx.j_+1; array[6].k_ = idx.k_+1;
  array[7].i_ = idx.i_;   array[7].j_ = idx.j_+1; array[7].k_ = idx.k_+1;
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::get_edges(typename Edge::array_type &, typename Face::index_type) const
{}
template <class Basis>
void 
MaskedLatVolMesh<Basis>::get_edges(typename Edge::array_type &, typename Cell::index_type) const
{}
template <class Basis>
void 
MaskedLatVolMesh<Basis>::get_faces(typename Face::array_type &, typename Cell::index_type) const
{}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Node::index_type idx) const
{
  unsigned i = idx.i_, j = idx.j_, k = idx.k_;
  return (check_valid(i  ,j  ,k  ) ||
	  check_valid(i-1,j  ,k  ) ||
	  check_valid(i  ,j-1,k  ) ||
	  check_valid(i  ,j  ,k-1) ||
	  check_valid(i-1,j-1,k  ) ||
	  check_valid(i-1,j  ,k-1) ||
	  check_valid(i  ,j-1,k-1) ||
	  check_valid(i-1,j-1,k-1));	  
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Edge::index_type idx) const
{
  
  bool val = false;
  if (idx.dir_ == 0)
  {
    val =  ((idx.i_ < min_i_ + ni_ - 1) &&
	     (check_valid(idx.i_  ,idx.j_  ,idx.k_  ) ||
	      check_valid(idx.i_  ,idx.j_-1,idx.k_  ) ||
	      check_valid(idx.i_  ,idx.j_  ,idx.k_-1) ||
	      check_valid(idx.i_  ,idx.j_-1,idx.k_-1)));
  }
  if (idx.dir_ == 1)
  {
    val =   ((idx.j_ < min_j_ + nj_ - 1) &&
	      (check_valid(idx.i_  ,idx.j_  ,idx.k_) ||
	       check_valid(idx.i_-1,idx.j_  ,idx.k_) ||
	       check_valid(idx.i_  ,idx.j_  ,idx.k_-1) ||
	       check_valid(idx.i_-1,idx.j_  ,idx.k_-1)));
  }
  if (idx.dir_ == 2)
  { 
    val =  ((idx.k_ < min_k_ + nk_ - 1) &&
	    (check_valid(idx.i_  ,idx.j_,  idx.k_) ||
	     check_valid(idx.i_-1,idx.j_,  idx.k_) ||
	     check_valid(idx.i_  ,idx.j_-1,idx.k_) ||
	     check_valid(idx.i_-1,idx.j_-1,idx.k_)));
  }
  return val;
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Face::index_type idx) const
{
  if (idx.dir_ == 0)
  {
    return (idx.i_ < min_i_ + ni_ - 1 &&
	    idx.j_ < min_j_ + nj_ - 1 &&
	    (check_valid(idx.i_,idx.j_,idx.k_) ||
	     check_valid(idx.i_,idx.j_,idx.k_-1)));
  }
  if (idx.dir_ == 1)
  {
    return (idx.j_ < min_j_ + nj_ - 1 &&
	    idx.k_ < min_k_ + nk_ - 1 &&
	    (check_valid(idx.i_,idx.j_,idx.k_) ||
	     check_valid(idx.i_-1,idx.j_,idx.k_)));
  }
  if (idx.dir_ == 2)
  {
    return (idx.i_ < min_i_ + ni_ - 1 &&
	    idx.k_ < min_k_ + nk_ - 1 &&
	    (check_valid(idx.i_,idx.j_,idx.k_) ||
	     check_valid(idx.i_,idx.j_-1,idx.k_)));
  }

  return false;

}

template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Cell::index_type i) const
{
  return check_valid(i.i_,i.j_,i.k_);
}



template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Node::iterator i) const
{
  return check_valid(*i);
}

template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Edge::iterator i) const
{
  return check_valid(*i);
}

template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Face::iterator i) const
{
  return check_valid(*i);
}

template <class Basis>
bool
MaskedLatVolMesh<Basis>::check_valid(typename MaskedLatVolMesh<Basis>::Cell::iterator i) const
{
  return check_valid(i.i_,i.j_,i.k_);
}


//! This function updates the missing node, edge, and face count
//! when masking or unmasking one cell. 
//! Returns true if nodes, edges, or faces count is changed
template <class Basis>
bool
MaskedLatVolMesh<Basis>::update_count(typename MaskedLatVolMesh<Basis>::Cell::index_type c, 
				      bool masking)
{
  synchronized_ &= ~NODES_E;
  const bool i0 = (c.i_ > min_i_) && check_valid(c.i_-1, c.j_, c.k_);
  const bool j0 = (c.j_ > min_j_) && check_valid(c.i_, c.j_-1, c.k_);
  const bool k0 = (c.k_ > min_k_) && check_valid(c.i_, c.j_, c.k_-1);
  const bool i1 = (c.i_ < min_i_+ni_-1) && check_valid(c.i_+1, c.j_, c.k_);
  const bool j1 = (c.j_ < min_j_+nj_-1) && check_valid(c.i_, c.j_+1, c.k_);
  const bool k1 = (c.k_ < min_k_+nk_-1) && check_valid(c.i_, c.j_, c.k_+1);

  // These counts are the number of nodes, edges, faces that exist
  // ONLY from the presence of this cell, not because of the contribution 
  // of neighboring cells.
  const unsigned faces = (i0?0:1)+(i1?0:1)+(j0?0:1)+(j1?0:1)+(k0?0:1)+(k1?0:1);
  unsigned int   nodes = 0;
  unsigned int   edges = 0;

  if (faces == 6) {  
	nodes = 8; 
	edges = 12;
  } 
  else {
	if (faces == 5)	{ 
	  nodes = 4; edges = 8;
	}
	else { 
	  if (faces == 1 || faces == 0)	{ 
		nodes = 0; edges = 0; 
	  }
	  else { 
		if(faces == 4) {
		  if((i0 == i1) && (j0 == j1) && (k0 == k1)) {
			nodes = 0;
			edges = 4;
		  }
		  else {
			nodes = 2;
			edges = 5;
		  }
		}
		else {
		  if(faces == 3) {
			if((i0!=i1)&&(j0!=j1)&&(k0!=k1)) {
			  nodes = 1;
			  edges = 3;
			}
			else {
			  nodes = 0;
			  nodes = 2;
			}
		  }
		  else {
			if(faces == 2) {
			  if((i0 == i1) && (j0 == j1) && (k0 == k1)) {
				nodes = 0;
				edges = 0;
			  }
			  else {
				nodes = 0;
				edges = 1;
			  }
			}
		  }
		}
	  }
	}
  }

  // These nodes, edges, faces are being implicitly removed from the mesh
  // by the removal of this cell.
  if (masking)
  {
    masked_nodes_count_ += nodes;
    masked_edges_count_ += edges;
    masked_faces_count_ += faces;
  }
  // These ndoes, edges, & faces are being implicitly added back into the mesh
  // because this cell is being added back in
  else 
  {
    masked_nodes_count_ -= nodes;
    masked_edges_count_ -= edges;
    masked_faces_count_ -= faces;
  }

  return (faces == 0);
}

  


template <class Basis>
BBox
MaskedLatVolMesh<Basis>::get_bounding_box() const
{
  // TODO:  return bounding box of valid cells only
  return LatVolMesh<Basis>::get_bounding_box();
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Node::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center
    (result, typename LatVolMesh<Basis>::Node::index_type(this,idx.i_,idx.j_,idx.k_));
}



template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Edge::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center(result, typename LatVolMesh<Basis>::Edge::index_type(unsigned(idx))); 
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Face::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center(result, typename LatVolMesh<Basis>::Face::index_type(unsigned(idx))); 
}


template <class Basis>
void
MaskedLatVolMesh<Basis>::get_center(Point &result, const typename Cell::index_type &idx) const
{
  ASSERT(check_valid(idx));
  LatVolMesh<Basis>::get_center
    (result,typename LatVolMesh<Basis>::Cell::index_type(this,idx.i_,idx.j_,idx.k_));
}


template <class Basis>
bool
MaskedLatVolMesh<Basis>::locate(typename Node::index_type &idx, const Point &p)
{
  typename LatVolMesh<Basis>::Node::index_type i(this,idx.i_,idx.j_,idx.k_);
  bool lat = LatVolMesh<Basis>::locate(i,p);
  idx.i_ = i.i_; idx.j_ = i.j_; idx.k_ = i.k_; idx.mesh_ = this;
  if (lat && check_valid(idx))
  {
    return true;
  }
  else return false;    
}

template <class Basis>
bool
MaskedLatVolMesh<Basis>::locate(typename Cell::index_type &idx, const Point &p)
{
  typename LatVolMesh<Basis>::Cell::index_type i(this,idx.i_,idx.j_,idx.k_);
  bool lat = LatVolMesh<Basis>::locate(i,p);
  idx.i_ = i.i_; idx.j_ = i.j_; idx.k_ = i.k_; idx.mesh_ = this;
  if (lat && check_valid(idx))
  {
    return true;
  }
  else return false;
}

template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Node::index_type idx) const
{
  ASSERT(check_valid(idx));
  typename LatVolMesh<Basis>::Node::index_type i(this,idx.i_,idx.j_,idx.k_);
  return LatVolMesh<Basis>::get_size(i);
}

template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Edge::index_type idx) const
{
  ASSERT(check_valid(idx));
  return LatVolMesh<Basis>::get_size(typename LatVolMesh<Basis>::Edge::index_type(unsigned(idx)));
}

template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Face::index_type idx) const
{
  ASSERT(check_valid(idx));
  return LatVolMesh<Basis>::get_size(typename LatVolMesh<Basis>::Face::index_type(unsigned(idx)));
}

template <class Basis>
double
MaskedLatVolMesh<Basis>::get_size(typename Cell::index_type idx) const
{
  ASSERT(check_valid(idx));
  typename LatVolMesh<Basis>::Cell::index_type i(this,idx.i_,idx.j_,idx.k_);
  return LatVolMesh<Basis>::get_size(i);
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::mask_cell(typename Cell::index_type idx)
{
  update_count(idx,true);
  masked_cells_.insert(unsigned(idx));
}

template <class Basis>
void
MaskedLatVolMesh<Basis>::unmask_cell(typename Cell::index_type idx)
{
  update_count(idx,false);
  masked_cells_.erase(unsigned(idx));
}


template <class Basis>
unsigned
MaskedLatVolMesh<Basis>::num_masked_nodes() const
{
  return masked_nodes_count_;
}

template <class Basis>
unsigned
MaskedLatVolMesh<Basis>::num_masked_edges() const
{
  return masked_edges_count_;
}

template <class Basis>
unsigned
MaskedLatVolMesh<Basis>::num_masked_faces() const
{
  return masked_faces_count_;
}

template <class Basis>
unsigned
MaskedLatVolMesh<Basis>::num_masked_cells() const
{
  return masked_cells_.size();
}

template <class Basis>
void 
MaskedLatVolMesh<Basis>::
get_neighbors_stencil(vector<pair<bool,typename Cell::index_type> > &nbrs, 
		      typename Cell::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_ + 1); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_ + 1); j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_ + 1); i++)
		if (i != int(idx.i_) || j != int(idx.j_) || k != int(idx.k_))
		  if (i >= int(min_i_) && j >= int(min_j_) && k >= int(min_k_) &&
			  i <= int(min_i_+ni_)-1 && j <= int(min_j_+nj_)-1 && 
			  i <= int(min_k_+nk_)-1 && check_valid(i,j,k))
			nbrs.push_back(make_pair(true,typename Cell::index_type(this,i,j,k)));
		  else
			nbrs.push_back(make_pair(false,typename Cell::index_type(0,0,0,0)));
}


template <class Basis>
void 
MaskedLatVolMesh<Basis>::
get_neighbors_stencil(vector<pair<bool,typename Node::index_type> > &nbrs, 
		      typename Node::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_) + 1; k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_) + 1; j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_) + 1; i++)
	if (i != int(idx.i_) || j != int(idx.j_) || k != int(idx.k_))
	  if (i >= int(min_i_) && j >= int(min_j_) && k >= int(min_k_) &&
	      i <= int(min_i_+ni_) && j <= int(min_j_+nj_) &&
	      i <= int(min_k_+nk_) &&
	      check_valid(typename Node::index_type(this,i,j,k)))
	    nbrs.push_back(make_pair(true,typename Node::index_type(this,i,j,k)));
	  else
	    nbrs.push_back(make_pair(false,typename Node::index_type(0,0,0,0)));
}


template <class Basis>
void 
MaskedLatVolMesh<Basis>::
get_neighbors_stencil(vector<pair<bool,typename Cell::index_type> > &nbrs, 
		      typename Node::index_type idx) const
{
  nbrs.clear();
  for (int k = idx.k_ - 1; k <= int(idx.k_); k++)
    for (int j = idx.j_ - 1; j <= int(idx.j_); j++)
      for (int i = idx.i_ - 1; i <= int(idx.i_); i++)
		if (i >= int(min_i_) && j >= int(min_j_) && k >= int(min_k_) &&
			i <= int(min_i_+ni_)-1 && j <= int(min_j_+nj_)-1 &&
			i <= int(min_k_+nk_)-1 && check_valid(i,j,k))
		  nbrs.push_back(make_pair(true,typename Cell::index_type(this,i,j,k)));
		else
		  nbrs.push_back(make_pair(false,typename Cell::index_type(0,0,0,0)));
}

    
template <class Basis>
unsigned int
MaskedLatVolMesh<Basis>::get_sequential_node_index(const typename Node::index_type idx)
{
  node_lock_.lock();
  if (synchronized_ & NODES_E) {
    node_lock_.unlock();
  }

  nodes_.clear();
  int i = 0;
  typename Node::iterator node, nend;
  begin(node);
  end(nend);
  while (node != nend) {
    nodes_[*node] = i++;
    ++node;
  }
  synchronized_ |= NODES_E;
  node_lock_.unlock();

  return nodes_[idx];
}


template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::NodeIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    Pio(stream, n.k_);
    stream.end_cheap_delim();
}

template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::CellIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    Pio(stream, n.k_);
    stream.end_cheap_delim();
}

template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::EdgeIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    Pio(stream, n.k_);
    stream.end_cheap_delim();
}

template <class Basis>
void
Pio(Piostream& stream, typename MaskedLatVolMesh<Basis>::FaceIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    Pio(stream, n.k_);
    stream.end_cheap_delim();
}

template <class Basis>
const string 
find_type_name(typename MaskedLatVolMesh<Basis>::NodeIndex *)
{
  static string name = MaskedLatVolMesh<Basis>::type_name(-1) + "::NodeIndex";
  return name;
}

template <class Basis>
const string 
find_type_name(typename MaskedLatVolMesh<Basis>::CellIndex *)
{
  static string name = MaskedLatVolMesh<Basis>::type_name(-1) + "::CellIndex";
  return name;
}


#define MASKED_LAT_VOL_MESH_VERSION 1

template <class Basis>
void
MaskedLatVolMesh<Basis>::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), MASKED_LAT_VOL_MESH_VERSION);
  
  LatVolMesh<Basis>::io(stream);

  // IO data members, in order
  vector<unsigned> masked_vec(masked_cells_.begin(), 
			      masked_cells_.end());
  Pio(stream, masked_vec);
  if (stream.reading())
    {
      masked_cells_.clear();
      masked_cells_.insert(masked_vec.begin(), masked_vec.end());
    }

  stream.end_class();
}

template <class Basis>
const string
MaskedLatVolMesh<Basis>::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "MaskedLatVolMesh";
  return name;
}

template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
}

template <class Basis>
const TypeDescription*
get_type_description(MaskedLatVolMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(MaskedLatVolMesh<Basis>::type_name(0), subs,
				string(__FILE__),
				"SCIRun");
  }
  return td;
}


template <class Basis>
const TypeDescription* 
MaskedLatVolMesh<Basis>::node_index_type_description()
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *me = 
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::NodeIndex",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription* 
MaskedLatVolMesh<Basis>::cell_index_type_description()
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *me = 
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::CellIndex",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class Basis>
const TypeDescription*
MaskedLatVolMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me = 
      SCIRun::get_type_description((MaskedLatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_MaskedLatVolMesh_h
