#ifndef SCI_Wangxl_Datatypes_Mesh_DCell_h
#define SCI_Wangxl_Datatypes_Mesh_DCell_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCell.h>

namespace Wangxl {

using namespace SCIRun;

class DVertex;

class DCell : public VMCell
{

public:

  DCell() : VMCell() {}

  DCell(DVertex* v0, DVertex* v1, DVertex* v2, DVertex* v3)
    : VMCell((VMVertex*)v0, (VMVertex*)v1, (VMVertex*)v2, (VMVertex*)v3) {}
    
  DCell(DVertex* v0, DVertex* v1, DVertex* v2, DVertex* v3,
	DCell* n0, DCell* n1, DCell* n2, DCell* n3)
    : VMCell((VMVertex*)v0, (VMVertex*)v1, (VMVertex*)v2, (VMVertex*)v3, (VMCell*)n0, (VMCell*)n1, (VMCell*)n2, (VMCell*)n3) {}

  // Vertex access functions
  DVertex* vertex(int i) const
  {
    return  ((DVertex*)(VMCell::vertex(i)));
  }
    
  bool has_vertex(const DVertex* v) const
  {
    return (VMCell::has_vertex((VMVertex*)v) );
  }
    
  bool has_vertex(const DVertex* v, int & i) const
  {
    return VMCell::has_vertex((VMVertex*)v, i);
  }

  int index(const DVertex* v) const
  {
    return VMCell::index((VMVertex*)v);
  }

  //ACCESS FUNCTIONS
  DCell* neighbor(int i) const
  {
    return (DCell*)(VMCell::neighbor(i));
  }

  int index(DCell* c) const
  {
    return VMCell::index((VMCell*)c);
  }

  DVertex* mirror_vertex(int i) const
    {
      return (DVertex*) VMCell::mirror_vertex(i);
    }

  bool has_neighbor(DCell* c) const
  {
    return VMCell::has_neighbor((VMCell*)c);
  }

  bool has_neighbor(DCell* c, int& i) const
  {
    return VMCell::has_neighbor((VMCell*)c, i);
  }
 
 //Setting
  void set_vertices(DVertex* v0, DVertex* v1, DVertex* v2, DVertex* v3)
  {
    VMCell::set_vertices((VMVertex*)v0, (VMVertex*)v1, (VMVertex*)v2, (VMVertex*)v3);
  }
    
  void set_neighbors(DCell* n0, DCell* n1, DCell* n2, DCell* n3)
  {
    VMCell::set_neighbors((VMCell*)n0, (VMCell*)n1, (VMCell*)n2, (VMCell*)n3);
  }
    
  void set_vertex(int i, DVertex* v)
  {
    VMCell::set_vertex(i, (VMVertex*)v);
  }
    
  void set_neighbor(int i, DCell* n)
  {
    VMCell::set_neighbor(i, (VMCell*)n);
  }

  void set_domain(int d) { d_domain = d; }
  bool in_boundary() const { 
    if ( d_domain == 1 ) return true; 
    else return false;
  }
  bool is_test() const {
    if ( d_domain == 0 ) return true;
    else return false;
  }
  int get_domain() const { return d_domain; }

private:
  int d_domain; // 0: outside cells; 1: inside cells
};

}

#endif



