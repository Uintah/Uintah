#ifndef SCI_Wangxl_Datatypes_Mesh_VMVertexBase_h
#define SCI_Wangxl_Datatypes_Mesh_VMVertexBase_h

#include <iostream>
#include <Core/Geometry/Point.h>

namespace Wangxl {

using namespace SCIRun;

class VMVertexBase
{
public:
  
  // CONSTRUCTORS
  
  VMVertexBase() : d_p(), d_c(0)  { d_input = false; }
  
  VMVertexBase(const Point& p) : d_p(p), d_c(0) { d_input = false; }
  
  VMVertexBase(const Point& p, void* c) : d_p(p), d_c(c) { d_input = false; }
  
  VMVertexBase(void* c) : d_p(), d_c(c) { d_input = false; }

  // ACCES 

  const Point& point() const { return d_p; }
    
  void* cell() const { return d_c; }

  // SETTING

  void set_point(const Point& p) { d_p = p; }
    
  void set_cell(void* c) { d_c = c; }

  // CHECKING

  // the following trivial is_valid allows
  // the user of derived cell base classes 
  // to add their own purpose checking
  bool is_valid(bool, int ) const
  { 
    return cell() != 0;
  }

  void set_input(bool input ) { d_input = input; }
  bool is_input() { return d_input; }

private:
  Point d_p;
  void * d_c;

  bool d_input;
};

/*std::istream& operator >> (std::istream& is, VMVertexBase & v)
  // non combinatorial information. Default = point
{
  Point p;
  is >> p;
  v.set_point(p);
  return is;
}

std::ostream& operator<< (std::ostream& os,  VMVertexBase & v)
  // non combinatorial information. Default = point
{
  os << v.point();
  return os;
}
*/
}

#endif
