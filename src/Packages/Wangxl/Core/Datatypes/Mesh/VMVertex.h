#ifndef SCI_Wangxl_Datatypes_Mesh_VMVertex_h
#define SCI_Wangxl_Datatypes_Mesh_VMVertex_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMVertexBase.h>

namespace Wangxl {

using namespace SCIRun;

class VMCell;

class  VMVertex : public VMVertexBase
{
public:
  // CONSTRUCTORS

   VMVertex() : VMVertexBase() { set_order_of_creation(); }
    
   VMVertex(const Point& p) : VMVertexBase(p) { set_order_of_creation(); }
    
   VMVertex(const Point& p, VMCell* c) : VMVertexBase(p, c)
  { set_order_of_creation(); }

   VMVertex(VMCell * c) : VMVertexBase(c)
  { set_order_of_creation(); }

  // ACCESS

  VMCell* cell() const
  {
    return (VMCell *) (VMVertexBase::cell());
  }
    
  // SETTING

  void set_cell(VMCell* c)
  {
    VMVertexBase::set_cell(c);
  }

  // CHECKING

  // bool is_valid(bool verbose = false, int level = 0) const;

  // used for symbolic perturbation in remove_vertex for Delaunay
  // undocumented
  void set_order_of_creation()
  {
    static int nb=-1; 
    d_order_of_creation = ++nb;
  }

  int get_order_of_creation() const
  {
      return d_order_of_creation;
  }
  
bool is_valid(bool verbose, int level) const
{
  bool result = VMVertexBase::is_valid(verbose,level);// && cell()->has_vertex(this);
  if ( ! result ) {
    if ( verbose )
      std::cerr << "invalid vertex" << std::endl;
    assert(false);
  }
  return result;
}

private:
  int d_order_of_creation;
};

template < class VH>
class Vertex_mesh_compare_order_of_creation {
public:
  bool operator()(VH u, VH v) const {
    return u->get_order_of_creation() < v->get_order_of_creation();
  }
};

}

#endif

