#ifndef SCI_Wangxl_Datatypes_Mesh_DVertex_h
#define SCI_Wangxl_Datatypes_Mesh_DVertex_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMVertex.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DCell.h>

namespace Wangxl {

using namespace SCIRun;

class DVertex : public VMVertex
{
public:
 
  DVertex() : VMVertex() { /*d_input = false;*/ }

  DVertex(const Point& p) : VMVertex(p) { /*d_input = false;*/ }

  DVertex(const Point& p, DCell* c) : VMVertex(p, (VMCell*)c) { /*d_input = false;*/ }

  DVertex(DCell* c) : VMVertex(c) { /*d_input = false;*/ }

  void set_cell(DCell* c) { VMVertex::set_cell(c); }

  DCell* cell() const { return (DCell*) VMVertex::cell(); }

  /*  void set_input(bool input ) { d_input = input; }
      bool is_input() { return d_input; }
private:
bool d_input;*/
};

}

#endif











