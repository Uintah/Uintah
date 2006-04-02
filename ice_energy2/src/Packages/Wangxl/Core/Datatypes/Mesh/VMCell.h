#ifndef SCI_Wangxl_Datatypes_Mesh_VMCell_h
#define SCI_Wangxl_Datatypes_Mesh_VMCell_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMVertex.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCellBase.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Utilities.h>

namespace Wangxl {

using namespace SCIRun;

class VolumeMesh;
template<class VolumeMesh> class VMCellIterator;
template<class VolumeMesh> class VMFacetIterator;
template<class VolumeMesh> class VMEdgeIterator;
template<class VolumeMesh> class VMVertexIterator;

class VMCell : public VMCellBase
{
    static int ccw(int i)
    {
      return Utilities::ccw(i);
    } 

    static int cw(int i)
    {
      return Utilities::cw(i);
    } 

public:

  friend class VolumeMesh;
  friend class VMCellIterator<VolumeMesh>;
  friend class VMFacetIterator<VolumeMesh>;
  friend class VMEdgeIterator<VolumeMesh>;
  friend class VMVertexIterator<VolumeMesh>;

  VMCell() : VMCellBase(), d_conflict_flag(0) {}

  VMCell(VMCell* c) : VMCellBase(*c), d_conflict_flag(0) {}
    
  VMCell(VMVertex* v0, VMVertex* v1, VMVertex* v2, VMVertex* v3)
    :  VMCellBase(v0,v1,v2,v3), d_conflict_flag(0) {}

   VMCell(VMVertex* v0, VMVertex* v1, VMVertex* v2, VMVertex* v3,
	  VMCell* n0, VMCell* n1, VMCell* n2, VMCell* n3)
    :  VMCellBase(v0,v1,v2,v3,n0,n1,n2,n3), d_conflict_flag(0) {}

  // SETTING

  void set_vertex(int i, VMVertex* v)
  {
     VMCellBase::set_vertex(i,v);
  }

  void set_neighbor(int i, VMCell* n)
  {
     VMCellBase::set_neighbor(i,n);
  }

  void set_vertices(VMVertex* v0, VMVertex* v1, VMVertex* v2, VMVertex* v3)
  {
     VMCellBase::set_vertices(v0,v1,v2,v3);
  }

  void set_neighbors(VMCell* n0, VMCell* n1, VMCell* n2, VMCell* n3)
  {
     VMCellBase::set_neighbors(n0,n1,n2,n3);
  }

  // VERTEX ACCESS

  VMVertex* vertex(int i) const
  {
    return (VMVertex*) ( VMCellBase::vertex(i));
  } 

  bool has_vertex(const VMVertex* v) const
  {
    return  VMCellBase::has_vertex(v);
  }
    
  bool has_vertex(const VMVertex* v, int & i) const
  {
    return  VMCellBase::has_vertex(v,i);
  }
    
  int index(const VMVertex* v) const
  {
    return  VMCellBase::vertex_index(v);
  }

  // NEIGHBOR ACCESS

  VMCell* neighbor(int i) const
  {
    return (VMCell*)  VMCellBase::neighbor(i);
  }
    
  bool has_neighbor(const VMCell* n) const
  {
    return VMCellBase::has_neighbor(n);
  }
    
  bool has_neighbor(const VMCell* n, int & i) const
  {
    return  VMCellBase::has_neighbor(n,i);
  }
    
  int index(const VMCell* n) const
  {
    return  VMCellBase::cell_index(n);
  }

  int mirror_index(int i) const
  {
      assert ( i>=0 && i<4 );
      return neighbor(i)->index(this);
  }
      
  // CHECKING
    
  VMVertex* mirror_vertex(int i) const
  {
      return neighbor(i)->vertex(mirror_index(i));
  }

  //  bool is_valid(int dim = 3, bool verbose = false, int level = 0) const;
bool VMCell::is_valid(int dim, bool verbose, int level) const
{
  if ( ! VMCellBase::is_valid(verbose,level) )
    return false;

  switch (dim) {
  case -2:
  case -1:
    {
    if ( vertex(0) == 0 ) {
      if (verbose) std::cerr << "vertex 0 NULL" << std::endl;
      assert(false);
      return false;
    }
    vertex(0)->is_valid(verbose,level);
    if ( vertex(1) != 0 || vertex(2) != 0 || vertex(3) != 0 ) {
      if (verbose) std::cerr << "vertex 1,2 or 3 != NULL" << std::endl;
      assert(false);
      return false;
    }
    if ( neighbor(0) != 0 || neighbor(1) != 0 ||
	 neighbor(2) != 0 || neighbor(3) != 0 ) {
      if (verbose) std::cerr << "one neighbor != NULL" << std::endl;
      assert(false);
      return false;
    }
    break;
    }
  case 0:
    {
    if ( vertex(0) == 0 ) {
      if (verbose) std::cerr << "vertex 0 NULL" << std::endl;
      assert(false);
      return false;
    }
    vertex(0)->is_valid(verbose,level);
    if ( neighbor (0) == 0 ) {
      if (verbose) std::cerr << "neighbor 0 NULL" << std::endl;
      assert(false);
      return false;
    }
    if ( vertex(1) != 0 || vertex(2) != 0 || vertex(3) != 0 ) {
      if (verbose) std::cerr << "vertex 1, 2 or 3 != NULL" << std::endl;
      assert(false);
      return false;
    }
    if ( neighbor(1) != 0 || neighbor(2) != 0 || neighbor(3) != 0 ) {
      if (verbose) std::cerr << "neighbor 1, 2 or 3 != NULL" << std::endl;
      assert(false);
      return false;
    }
    
    if ( ! neighbor(0)->has_vertex(vertex(0)) ) {
      if (verbose) std::cerr << "neighbor 0 does not have vertex 0" << std::endl;
      assert(false);
      return false;
    }
    break;
    }
  case 1:
    {
    VMVertex* v0 = vertex(0); 
    VMVertex* v1 = vertex(1);
    VMCell* n0 = neighbor(0); 
    VMCell* n1 = neighbor(1);
    
    if ( v0 == 0 || v1 == 0 ) {
      if (verbose) std::cerr << "vertex 0 or 1 NULL" << std::endl;
      assert(false);
      return false;
    }
    vertex(0)->is_valid(verbose,level);
    vertex(1)->is_valid(verbose,level);
    if ( n0 == 0 || n1 == 0 ) {
      if (verbose) std::cerr << "neighbor 0 or 1 NULL" << std::endl;
      assert(false);
      return false;
    }
    if ( vertex(2) != 0 || vertex(3) != 0 ) {
      if (verbose) std::cerr << "vertex 2 or 3 != NULL" << std::endl;
      assert(false);
      return false;
    }
    if ( neighbor(2) != 0 || neighbor(3) != 0 ) {
      if (verbose) std::cerr << "neighbor 2 or 3 != NULL" << std::endl;
      assert(false);
      return false;
    }

    if ( v0 !=  n1->vertex(1) ) {
      if (verbose) std::cerr << "neighbor 1 does not have vertex 0 as vertex 1"
			     << std::endl;
      assert(false);
      return false;
    }
    if ( v1 != n0->vertex(0) ) {
      if (verbose) std::cerr << "neighbor 0 does not have vertex 1 as vertex 0"
			     << std::endl;
      assert(false);
      return false;
    }
    
    if ( this != n0->neighbor(1) ) {
      if (verbose) std::cerr << "neighbor 0 does not have this as neighbor 1" 
			     << std::endl;
      assert(false);
      return false;
    }
    if ( this != n1->neighbor(0) ) {
      if (verbose) std::cerr << "neighbor 1 does not have this as neighbor 0" 
			     << std::endl;
      assert(false);
      return false;
    }
    break;
    }
  case 2:
    {
    if ( vertex(0) == 0 || vertex(1) == 0 || vertex(2) == 0 ) {
      if (verbose) std::cerr << "vertex 0, 1, or 2 NULL" << std::endl;
      assert(false);
      return false;
    }
    vertex(0)->is_valid(verbose,level);
    vertex(1)->is_valid(verbose,level);
    vertex(2)->is_valid(verbose,level);
    if ( vertex(3) != 0 ) {
      if (verbose) std::cerr << "vertex 3 != NULL" << std::endl;
      assert(false);
      return false;
    }
    if ( neighbor(3) != 0 ) {
      if (verbose) std::cerr << "neighbor 3 != NULL" << std::endl;
      assert(false);
      return false;
    }

    int in;
    VMCell* n;
    for(int i = 0; i < 3; i++) {
      n = neighbor(i);
      if ( n == 0 ) {
	if (verbose) std::cerr << "neighbor " << i << " NULL" << std::endl;
	assert(false);
	return false;
      }
      if ( ! n->has_vertex(vertex(cw(i)),in ) ) {
	if (verbose)
	  std::cerr << "vertex " << cw(i) 
		    << " not vertex of neighbor " << i << std::endl;
	assert(false);
	return false;
      }
      in = cw(in); 
      if ( this != n->neighbor(in) ) {
	if (verbose)
	  std::cerr << "neighbor " << i
		    << " does not have this as neighbor " 
		    << in << std::endl;
	assert(false);
	return false;
      }
      if ( vertex(ccw(i)) != n->vertex(cw(in)) ) {
	if (verbose) std::cerr << "vertex " << ccw(i)
			       << " is not vertex " << cw(in) 
			       << " of neighbor " << i << std::endl;
	assert(false);
	return false;
      }
    }
    break;
    }
  case 3:
    {
    int i;
    for(i = 0; i < 4; i++) {
      if ( vertex(i) == 0 ) {
	if (verbose) std::cerr << "vertex " << i << " NULL" << std::endl;
	assert(false);
	return false;
      }
      vertex(i)->is_valid(verbose,level);
    }
    
    for(i = 0; i < 4; i++) {
      VMCell* n = neighbor(i);
      if ( n == 0 ) {
	if (verbose)
	  std::cerr << "neighbor " << i << " NULL" << std::endl;
	assert(false);
	return false;
      }
      
      int in;
      if ( ! n->has_neighbor(this,in) ) {
	if (verbose) error_neighbor(n,i,in); 
	assert(false);
	return false;
      }
      
      int j1n,j2n,j3n;
      if ( ! n->has_vertex(vertex((i+1)&3),j1n) ) {
	if (verbose) { std::cerr << "vertex " << ((i+1)&3)
				 << " not vertex of neighbor " 
				 << i << std::endl; }
	assert(false);
	return false;
      }
      if ( ! n->has_vertex(vertex((i+2)&3),j2n) ) {
	if (verbose) { std::cerr << "vertex " << ((i+2)&3)
				 << " not vertex of neighbor " 
				 << i << std::endl; }
	assert(false);
	return false;
      }
      if ( ! n->has_vertex(vertex((i+3)&3),j3n) ) {
	if (verbose) { std::cerr << "vertex " << ((i+3)&3)
				 << " not vertex of neighbor "
				 << i << std::endl; }
	assert(false);
	return false;
      }
      
      if ( in+j1n+j2n+j3n != 6) {
	if (verbose) { std::cerr << "sum of the indices != 6 " 
				 << std::endl; }
	assert(false);
	return false;
      }
      
      // tests whether the orientations of this and n are consistent
      if ( ((i+in)&1) == 0 ) { // i and in have the same parity
	if ( j1n == ((in+1)&3) ) {
	  if ( ( j2n != ((in+3)&3) ) || ( j3n != ((in+2)&3) ) ) {
	    if (verbose) { 
	      error_orient(n,i);
	    }
	    assert(false);
	    return false;
	  }
	}
	if ( j1n == ((in+2)&3) ) {
	  if ( ( j2n != ((in+1)&3) ) || ( j3n != ((in+3)&3) ) ) {
	    if (verbose) { 
	      error_orient(n,i);
	    }
	    assert(false);
	    return false;
	  }
	}
	if ( j1n == ((in+3)&3) ) {
	  if ( ( j2n != ((in+2)&3) ) || ( j3n != ((in+1)&3) ) ) {
	    if (verbose) { 
	      error_orient(n,i);
	    }
	    assert(false);
	    return false;
	  }
	}
      }
      else { // i and in do not have the same parity
	if ( j1n == ((in+1)&3) ) {
	  if ( ( j2n != ((in+2)&3) ) || ( j3n != ((in+3)&3) ) ) {
	    if (verbose) { 
	      error_orient(n,i);
	    }
	    assert(false);
	    return false;
	  }
	}
	if ( j1n == ((in+2)&3) ) {
	  if ( ( j2n != ((in+3)&3) ) || ( j3n != ((in+1)&3) ) ) {
	    if (verbose) { 
	      error_orient(n,i);
	    }
	    assert(false);
	    return false;
	  }
	}
	if ( j1n == ((in+3)&3) ) {
	  if ( ( j2n != ((in+1)&3) ) || ( j3n != ((in+2)&3) ) ) {
	    if (verbose) { 
	      error_orient(n,i);
	    }
	    assert(false);
	    return false;
	  }
	}
      }
    } // end looking at neighbors
    }
  } // end switch
  return true;
} // end is_valid

private:

  // to maintain the list of cells
  VMCell* d_previous;
  VMCell* d_next;
  int d_conflict_flag;

  void set_conflict_flag(int f)
  {
    d_conflict_flag = f;
  }

  int get_conflict_flag() const
  {
    return d_conflict_flag;
  }

  void error_orient( VMCell * , int i ) const
  {
    std::cerr << " pb orientation with neighbor " << i << std::endl;
  }

  void error_neighbor( VMCell* , int , int ) const
  {
    std::cerr << "neighbor of c has not c as neighbor" << std::endl;
  }
};

}
#endif



