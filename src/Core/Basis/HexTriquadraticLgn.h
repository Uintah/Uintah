//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : HexTriquadraticLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 3 2004

#if !defined(HexTriquadraticLgn_h)
#define HexTriquadraticLgn_h

#include <HexTrilinearLgn.h>

namespace SCIRun {

template <class T>
class HexTriquadraticLgn : public HexApprox<T> {
public:
  typedef T value_type;

  HexTriquadraticLgn() {}
  virtual ~HexTriquadraticLgn() {}

  int polynomial_order() const { return 2; }

  // Value at coord
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];
    return (T)((-1 + x)*(-1 + y)*(-1 + z)*(-1 + 2*x + 2*y + 2*z)*cd.node0()
	       +x*(-1 + y)*(-1 + 2*x - 2*y - 2*z)*(-1 + z)*cd.node1()
	       -(x*y*(-3 + 2*x + 2*y - 2*z)*(-1 + z))*cd.node2()
	       -((-1 + x)*y*(-1 + z)*(1 + 2*x - 2*y + 2*z))*cd.node3()
	       -((-1 + x)*(-1 + y)*(1 + 2*x + 2*y - 2*z)*z)*cd.node4()
	       +x*(-1 + y)*(3 - 2*x + 2*y - 2*z)*z*cd.node5()
	       +x*y*z*(-5 + 2*x + 2*y + 2*z)*cd.node6()
	       +(-1 + x)*y*(3 + 2*x - 2*y - 2*z)*z*cd.node7()
	       -4*(-1 + x)*x*(-1 + y)*(-1 + z)*nodes_[cd.edge0_index()]
	       +4*x*(-1 +y)*y*(-1 + z)*nodes_[cd.edge1_index()]
	       +4*(-1 + x)*x*y*(-1 + z)*nodes_[cd.edge2_index()]
	       -4*(-1 + x)*(-1 + y)*y*(-1 + z)*nodes_[cd.edge3_index()]
	       -4*(-1 + x)*(-1 + y)*(-1 + z)*z*nodes_[cd.edge4_index()]
	       +4*x*(-1 + y)*(-1 + z)*z*nodes_[cd.edge5_index()]
	       +4*(-1 + x)*y*(-1 + z)*z*nodes_[cd.edge6_index()]
	       -4*x*y*(-1 + z)*z*nodes_[cd.edge7_index()]
	       +4*(-1 + x)*x*(-1 + y)*z*nodes_[cd.edge8_index()]
	       -4*x*(-1 + y)*y*z*nodes_[cd.edge9_index()]
	       -4*(-1 + x)*x*y*z*nodes_[cd.edge10_index()]
	       +4*(-1 + x)*(-1 + y)*y*z*nodes_[cd.edge11_index()]);
  }
  
  //! First derivative at coord.
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<double> &derivs) const
  {
    const double x=coords[0], y=coords[1], z=coords[2];

    derivs.size(3);

    derivs[0]=(-1 + y)*(-1 + z)*(-3 + 4*x + 2*y + 2*z)*cd.node0()
      -((-1 + y)*(-1 + z)*(1 - 4*x + 2*y + 2*z))*cd.node1()
      -(y*(-3 + 4*x + 2*y - 2*z)*(-1 + z))*cd.node2()
      +y*(1 - 4*x + 2*y - 2*z)*(-1 + z)*cd.node3()
      -((-1 + y)*(-1 + 4*x + 2*y - 2*z)*z)*cd.node4()
      +(-1 + y)*(3 - 4*x + 2*y - 2*z)*z*cd.node5()
      +y*z*(-5 + 4*x + 2*y + 2*z)*cd.node6()
      +y*(1 + 4*x - 2*y - 2*z)*z*cd.node7()
      -4*(-1 + 2*x)*(-1 + y)*(-1 + z)*nodes_[cd.edge0_index()]
      +4*(-1 + y)*y*(-1 + z)*nodes_[cd.edge1_index()]
      +4*(-1 + 2*x)*y*(-1 + z)*nodes_[cd.edge2_index()]
      +4*y*(-1 + y + z - y*z)*nodes_[cd.edge3_index()]
      +4*z*(-1 + y + z - y*z)*nodes_[cd.edge4_index()]
      +4*(-1 + y)*(-1 + z)*z*nodes_[cd.edge5_index()]
      +4*y*(-1 + z)*z*nodes_[cd.edge6_index()]
      -4*y*(-1 + z)*z*nodes_[cd.edge7_index()]
      +4*(-1 + 2*x)*(-1 + y)*z*nodes_[cd.edge8_index()]
      -4*(-1 + y)*y*z*nodes_[cd.edge9_index()]
      +4*(1 - 2*x)*y*z*nodes_[cd.edge10_index()]
      +4*(-1 + y)*y*z*nodes_[cd.edge11_index()];
    
    derivs[1]=(-1 + x)*(-1 + z)*(-3 + 2*x + 4*y + 2*z)*cd.node0()
      +x*(1 + 2*x - 4*y - 2*z)*(-1 + z)*cd.node1()
      -(x*(-3 + 2*x + 4*y - 2*z)*(-1 + z))*cd.node2()
      -((-1 + x)*(-1 + z)*(1 + 2*x - 4*y + 2*z))*cd.node3()
      -((-1 + x)*(-1 + 2*x + 4*y - 2*z)*z)*cd.node4()
      +x*(1 - 2*x + 4*y - 2*z)*z*cd.node5()
      +x*z*(-5 + 2*x + 4*y + 2*z)*cd.node6()
      +(-1 + x)*(3 + 2*x - 4*y - 2*z)*z*cd.node7()
      +4*x*(-1 + x + z - x*z)*nodes_[cd.edge0_index()]
      +4*x*(-1 + 2*y)*(-1 + z)*nodes_[cd.edge1_index()]
      +4*(-1 + x)*x*(-1 + z)*nodes_[cd.edge2_index()]
      -4*(-1 + x)*(-1 + 2*y)*(-1 + z)*nodes_[cd.edge3_index()]
      +4*z*(-1 + x + z - x*z)*nodes_[cd.edge4_index()]
      +4*x*(-1 + z)*z*nodes_[cd.edge5_index()]
      +4*(-1 + x)*(-1 + z)*z*nodes_[cd.edge6_index()]
      -4*x*(-1 + z)*z*nodes_[cd.edge7_index()]
      +4*(-1 + x)*x*z*nodes_[cd.edge8_index()]
      +4*x*(1 - 2*y)*z*nodes_[cd.edge9_index()]
      -4*(-1 + x)*x*z*nodes_[cd.edge10_index()]
      +4*(-1 + x)*(-1 + 2*y)*z*nodes_[cd.edge11_index()];
    
    derivs[2]=(-1 + x)*(-1 + y)*(-3 + 2*x + 2*y + 4*z)*cd.node0()
      +x*(-1 + y)*(1 + 2*x - 2*y - 4*z)*cd.node1()
      +x*y*(1 - 2*x - 2*y + 4*z)*cd.node2()
      -((-1 + x)*y*(-1 + 2*x - 2*y + 4*z))*cd.node3()
      -((-1 + x)*(-1 + y)*(1 + 2*x + 2*y - 4*z))*cd.node4()
      +x*(-1 + y)*(3 - 2*x + 2*y - 4*z)*cd.node5()
      +x*y*(-5 + 2*x + 2*y + 4*z)*cd.node6()
      (-1 + x)*y*(3 + 2*x - 2*y - 4*z)*cd.node7()
      +4*x*(-1 + x + y - x*y)*nodes_[cd.edge0_index()]
      +4*x*(-1 + y)*y*nodes_[cd.edge1_index()]
      +4*(-1 + x)*x*y*nodes_[cd.edge2_index()]
      +4*y*(-1 + x + y - x*y)*nodes_[cd.edge3_index()]
      -4*(-1 + x)*(-1 + y)*(-1 + 2*z)*nodes_[cd.edge4_index()]
      +4*x*(-1 + y)*(-1 + 2*z)*nodes_[cd.edge5_index()]
      +4*(-1 + x)*y*(-1 + 2*z)*nodes_[cd.edge6_index()]
      +4*x*y*(1 - 2*z)*nodes_[cd.edge7_index()]
      +4*(-1 + x)*x*(-1 + y)*nodes_[cd.edge8_index()]
      -4*x*(-1 + y)*y*nodes_[cd.edge9_index()]
      -4*(-1 + x)*x*y*nodes_[cd.edge10_index()]
      +4*(-1 + x)*(-1 + y)*y*nodes_[cd.edge11_index()];  
  }

  //! return the parametric coordinates for value within the element.
  //! iterative solution...
  template <class ElemData>
  void get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const  
      {
	HexLocate< HexTriquadraticLgn<T> > CL;
	CL.get_coords(this, coords, value, cd);
      };
    
  //! add a node value corresponding to edge
  void add_node_value(const T &p) { nodes_.push_back(p); }
 
  static  const string type_name(int n = -1);

  virtual void io (Piostream& str);

protected:
  double distance(const T&, const T&) const;

  //! Additional support values.

  //! Triquadratic Lagrangian only needs additional nodes stored for each edge
  //! in the topology.
  vector<T>          nodes_; 
};


template <class T>
const TypeDescription* get_type_description(HexTriquadraticLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(HexTriquadraticLgn<T>::type_name(0), subs, 
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class T>
const string
HexTriquadraticLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("HexTriquadraticLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int HEXTRIQUADRATICLGN_VERSION = 1;
template <class T>
void
HexTriquadraticLgn<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), HEXTRIQUADRATICLGN_VERSION);
  Pio(stream, nodes_);
  stream.end_class();
}

} //namespace SCIRun

#endif // HexTriquadraticLgn_h
