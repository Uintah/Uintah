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
//    File   : TetCubicHmt.h
//    Author : Frank B. Sachse
//    Date   : Nov 30 2004

#if !defined(TetCubicHmt_h)
#define TetCubicHmt_h

#include <TetLinearLgn.h>

namespace SCIRun {

  //! Class for handling of element of type tetrahedron with cubic hermitian interpolation
  template <class T>
    class TetCubicHmt : public TetApprox<T> {
  public:
    typedef T value_type;

    TetCubicHmt() {}
    virtual ~TetCubicHmt() {}

    int polynomial_order() const { return 3; }

    //! get value at parametric coordinate  
    template <class ElemData>
      T interpolate(const vector<double> &coords, const ElemData &cd) const
      {
	const double x=coords[0], y=coords[1], z=coords[2];  
	return (-3*x*x + 2*x*x*x - 3*y*y + 2*y*y*y + 
		(z-1)*(z-1)*(1 + 2*z))*cd.node0()
	  +x*(1 + x*x + x*(-2 + y) - y - z*z)*derivs_[cd.node0_index()][0]
	  +y*(-x*x + (-1 + y)*(-1 + y + z))*derivs_[cd.node0_index()][1]
	  +(-y*y + x*(-1 + z) + (z-1)*(z-1))*z*derivs_[cd.node0_index()][2]
	  +(3 - 2*x)*x*x*cd.node1()
	  +(-1 + x)*x*x*derivs_[cd.node1_index()][0]
	  +x*x*y*derivs_[cd.node1_index()][1]
	  -(x*(-1 + z)*z)*derivs_[cd.node1_index()][2]
	  +(3 - 2*y)*y*y*cd.node2()
	  -((-1 + x)*x*y)*derivs_[cd.node2_index()][0]
	  +(-1 + y)*y*y*derivs_[cd.node2_index()][1]
	  +y*y*z*derivs_[cd.node2_index()][2]
	  +(3 - 2*z)*z*z*cd.node3()
	  +x*z*z*derivs_[cd.node3_index()][0]
	  -((-1 + y)*y*z)*derivs_[cd.node3_index()][1]
	  +(-1 + z)*z*z*derivs_[cd.node3_index()][2];
      }
  
    //! get first derivative at parametric coordinate
    template <class ElemData>
      void derivate(const vector<double> &coords, const ElemData &cd, 
		    vector<double> &derivs) const
      {
	const double x=coords[0], y=coords[1], z=coords[2]; 
 
	derivs.size(3);

	derivs[0]=6*(-1 + x)*x*cd.node0()
	  +(1 + 3*x*x + 2*x*(-2 + y) - y - z*z)*derivs_[cd.node0_index()][0]
	  -2*x*y*derivs_[cd.node0_index()][1]*derivs_[cd.node0_index()][1]
	  +(-1 + z)*z*derivs_[cd.node0_index()][2]
	  -6*(-1 + x)*x*cd.node1()
	  +x*(-2 + 3*x)*derivs_[cd.node1_index()][0]
	  +2*x*y*derivs_[cd.node1_index()][1]
	  -((-1  z)*z)*derivs_[cd.node1_index()][2]
	  +(y - 2*x*y)*derivs_[cd.node2_index()][0]
	  +z*z*derivs_[cd.node3_index()][0];

	derivs[1]=6*(-1 + y)*y*cd.node0()
	  +(-1 + x)*x*derivs_[cd.node0_index()][0]
	  +(1 - x*x + 3*y*y + 2*y*(-2 + z) - z)*derivs_[cd.node0_index()][1]
	  -2*y*z*derivs_[cd.node0_index()][2]
	  +x*x*derivs_[cd.node1_index()][1]
	  -6*(-1 + y)*y*cd.node2()
	  -((-1 + x)*x)*derivs_[cd.node2_index()][0]
	  +y*(-2 + 3*y)*derivs_[cd.node2_index()][1]
	  +2*y*z*derivs_[cd.node2_index()][2]
	  +(z - 2*y*z)*derivs_[cd.node3_index()][1];

	derivs[2]=6*(-1 + z)*z*cd.node0()
	  -2*x*z*derivs_[cd.node0_index()][0]
	  +(-1 + y)*y*derivs_[cd.node0_index()][1]
	  +(1 - x - y*y - 4*z + 2*x*z + 3*z*z)*derivs_[cd.node0_index()][2]
	  +(x - 2*x*z)*derivs_[cd.node1_index()][2]
	  +y*y*derivs_[cd.node2_index()][2]
	  -6*(-1 + z)*z*cd.node3()
	  +2*x*z*derivs_[cd.node3_index()][0]
	  -((-1 + y)*y)*derivs_[cd.node3_index()][1]
	  +z*(-2 + 3*z)*derivs_[cd.node3_index()][2];
      }
  
    //! get parametric coordinate for value within the element
    //! iterative solution...
    template <class ElemData>
      void get_coords(vector<double> &coords, const T& value, 
		      const ElemData &cd) const;  

    //! add derivative values (dx, dy, dz) for nodes.
    void add_derivative(const T &p[3]) { derivs_.push_back(p); }

    static  const string type_name(int n = -1);

    virtual void io (Piostream& str);

  protected:
    //! support data (node data is elsewhere)

    //! Cubic Hermitian only needs additonal derivatives stored at each node
    //! in the topology.
    vector<T[3]>          derivs_; 

    double distance(const T&, const T&) const;
  };


  template <class T>
    const TypeDescription* get_type_description(TetCubicHmt<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(TetCubicHmt<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    TetCubicHmt<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("TetCubicHmt");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }


  template <class T>
    template <class ElemData>
    void 
    TetCubicHmt<T>::get_coords(vector<double> &coords, const T& value, 
			       const ElemData &cd) const
    {
    }

  const int TETCUBICHMT_VERSION = 1;
  template <class T>
    void
    TetCubicHmt<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), TETCUBICHMT_VERSION);
      Pio(stream, derivs_);
      stream.end_class();
    }

} //namespace SCIRun

#endif // TetCubicHmt_h
