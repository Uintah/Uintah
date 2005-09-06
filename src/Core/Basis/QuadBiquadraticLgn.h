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
//    File   : QuadBiquadraticLgn.h
//    Author : Marty Cole, Frank B. Sachse
//    Date   : 30 Nov 2004

#if !defined(QuadBiquadraticLgn_h)
#define QuadBiquadraticLgn_h

#include <QuadBilinearLgn.h>

namespace SCIRun {

template <class T>
class QuadBiquadraticLgn : public QuadApprox<T>
{
public:
  typedef T value_type;

  QuadBiquadraticLgn() {}
  virtual ~QuadBiquadraticLgn() {}
  
  int polynomial_order() const { return 2; }

  //! get first derivative at parametric coordinate 
  template <class ElemData>
  T interpolate(const vector<double> &coords, const ElemData &cd) const;
  {
    const double x=coords[0], y=coords[1];  
    return -((-1 + x)*(-1 + y)*(-1 + 2*x + 2*y))*cd.node0()
      -(x*(-1 + 2*x - 2*y)*(-1 +y))*cd.node1()
      +x*y*(-3 + 2*x + 2*y)*cd.node2()
      +(-1 + x)*(1 + 2*x - 2*y)*y*cd.node3()
      +4*(-1 + x)*x*(-1 + y)*nodes_[cd.edge0_index()]
      -4*x*(-1 + y)*y*nodes_[cd.edge1_index()]
      -4*(-1 + x)*x*y*nodes_[cd.edge2_index()]
      +4*(-1 + x)*(-1 + y)*y*nodes_[cd.edge3_index()];
  };
  
  //! get first derivative at parametric coordinate
  template <class ElemData>
  void derivate(const vector<double> &coords, const ElemData &cd, 
		vector<double> &derivs) const;
  {
    const double x=coords[0], y=coords[1];  

    derivs.size(2);
    
    derivs[0]=-((-1 + y)*(-3 + 4*x + 2*y))*cd.node0()
      -((-1 + 4*x - 2*y)*(-1 + y))*cd.node1()
      +y*(-3+ 4*x + 2*y)*cd.node2()
      +(-1 + 4*x - 2*y)*y*cd.node3()
      +4*(-1 + 2*x)*(-1 + y)*nodes_[cd.edge0_index()]
      -4*(-1 + y)*y*nodes_[cd.edge1_index()]
      +(4 - 8*x)*y*nodes_[cd.edge2_index()]
      +4*(-1 + y)*y*nodes_[cd.edge3_index()];
    
    derivs[1]=-((-1 + x)*(-3 + 2*x +4*y))*cd.node0()
      +x*(-1 - 2*x + 4*y)*cd.node1()
      +x*(-3 + 2*x + 4*y)*cd.node2()
      +(-1 + x)*(1 + 2*x -4*y)*cd.node3()
      +4*(-1 + x)*x*nodes_[cd.edge0_index()]
      +x*(4 -8*y)*nodes_[cd.edge1_index()]
      -4*(-1 + x)*x*nodes_[cd.edge2_index()]
      +4*(-1 + x)*(-1 +2*y)*nodes_[cd.edge3_index()];
  };  
  
  //! get parametric coordinate for value within the element
  //! iterative solution...
  template <class ElemData>
  void get_coords(vector<double> &coords, const T& value, 
		  const ElemData &cd) const
    {
    }  

  //! add a node value corresponding to edge
  void add_node(const T &p) { nodes_.push_back(p); }

  static  const string type_name(int n = -1);
  virtual void io (Piostream& str);
  
protected:

  double distance(const T&, const T&) const;

  //! Additional support values.

  //! Quadratic Lagrangian only needs additional nodes stored for each edge
  //! in the topology.
  vector<T>          nodes_; 
};


template <class T>
const TypeDescription* get_type_description(QuadBiquadraticLgn<T> *)
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *sub = SCIRun::get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(QuadBiquadraticLgn<T>::type_name(0), subs, 
				string(__FILE__),
				"SCIRun");
  }
  return td;
}

template <class T>
const string
QuadBiquadraticLgn<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("QuadBiquadraticLgn");
    return nm;
  } else {
    return find_type_name((T *)0);
  }
}


const int QUADBIQUADRATICLGN_VERSION = 1;
template <class T>
void
QuadBiquadraticLgn<T>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), QUAD_BIQUADRATIC_LGN_VERSION);
  Pio(stream, nodes_);
  stream.end_class();
}

} //namespace SCIRun

#endif // QuadBiquadraticLgn_h
