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
//    File   : QuadBicubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(QuadBicubicHmt_h)
#define QuadBicubicHmt_h

#include <QuadBilinearLgn.h>

namespace SCIRun {

  
  //! Class for handling of element of type quad with bicubic hermitian interpolation
  template <class T>
    class QuadBicubicHmt : public QuadApprox, public QuadGaussian2<double>
  {
  public:
    typedef T value_type;

    QuadBicubicHmt() {}
    virtual ~QuadBicubicHmt() {}
  
    int polynomial_order() const { return 3; }

    //! get value at parametric coordinate 
    template <class CellData>
      T interpolate(const vector<double> &coords, const CellData &cd) const
      {
	const double x=coords[0], y=coords[1];  
	return
	  -((-1 + x)*(-1 + y)*(-1 - x + 2*x*x - y + 2*y*y))*cd.node0()
	  -((x-1)*(x-1)*x*(-1 + y))*derivs_[cd.node0_index()][0]
	  -((-1 + x)*(y-1)*(y-1)*y)*derivs_[cd.node0_index()][1]
	  +x*(-1 + y)*(-3*x + 2*x*x + y*(-1 + 2*y))*cd.node1()
	  +x*x*(-1 + x + y - x*y)*derivs_[cd.node1_index()][0]
	  +x*(y-1)*(y-1)*y*derivs_[cd.node1_index()][1]
	  +x*y*(-1 + 3*x - 2*x*x + 3*y - 2*y*y)*cd.node2()
	  +(-1 + x)*x*x*y*derivs_[cd.node2_index()][0]
	  +x*(-1 + y)*y*y*derivs_[cd.node2_index()][1]
	  +(-1 + x)*y*(-x + 2*x*x + y*(-3 + 2*y))*cd.node3()
	  +(x-1)*(x-1)*x*y*derivs_[cd.node3_index()][0]
	  +y*y*(-1 + x + y - x*y)*derivs_[cd.node3_index()][1];
      }
  
    //! get first derivative at parametric coordinate
    template <class CellData>
      void derivate(const vector<double> &coords, const CellData &cd, 
		    vector<double> &derivs) const
      {
	const double x=coords[0], y=coords[1];  

	derivs.size(2);

	derivs[0]=-((-1 + y)*(-6*x + 6*x*x + y*(-1 + 2*y)))*cd.node0()
	  -((1 - 4*x + 3*x*x)*(-1 + y))*derivs_[cd.node0_index()][0]
	  -((y-1)*(y-1)*y)*derivs_[cd.node0_index()][1]*derivs_[cd.node0_index()][1]
	  +(-1 + y)*(-6*x + 6*x*x + y*(-1 + 2*y))*cd.node1()
	  -(x*(-2 + 3*x)*(-1 + y))*derivs_[cd.node1_index()][0]
	  +(y-1)*(y-1)*y*derivs_[cd.node1_index()][1]
	  +y*(-1 + 6*x - 6*x*x + 3*y - 2*y*y)*cd.node2()
	  +x*(-2 + 3*x)*y*derivs_[cd.node2_index()][0]
	  +(-1 + y)*y*y*derivs_[cd.node2_index()][1]
	  +y*(1 - 6*x + 6*x*x - 3*y + 2*y*y)*cd.node3()
	  +(1 - 4*x + 3*x*x)*y*derivs_[cd.node3_index()][0]
	  -((-1 + y)*y*y)*derivs_[cd.node3_index()][1];
	
	derivs[1]=-((-1 + x)*(-x + 2*x*x + 6*(-1 + y)*y))*cd.node0()
	  -((x-1)*(x-1)*x)*derivs_[cd.node0_index()][0]
	  -((-1 + x)*(1 - 4*y + 3*y*y))*derivs_[cd.node0_index()][1]
	  +x*(1 - 3*x + 2*x*x - 6*y + 6*y*y)*cd.node1()
	  -((-1 + x)*x*x)*derivs_[cd.node1_index()][0]
	  +x*(1 - 4*y + 3*y*y)*derivs_[cd.node1_index()][1]
	  +x*(-1 + 3*x - 2*x*x + 6*y - 6*y*y)*cd.node2()
	  (-1 + x)*x*x*derivs_[cd.node2_index()][0]
	  +x*y*(-2 + 3*y)*derivs_[cd.node2_index()][1]
	  (-1 + x)*(-x + 2*x*x + 6*(-1 + y)*y)*cd.node3()
	  +(x-1)*(x-1)*x*derivs_[cd.node3_index()][0]
	  -((-1 + x)*y*(-2 + 3*y))*derivs_[cd.node3_index()][1];
      };
  
    //! get parametric coordinate for value within the element
    template <class CellData>
      void get_coords(vector<double> &coords, const T& value, 
		      const CellData &cd) const
      {
      }  

    //! add derivative values (dx, dy) for nodes.
    void add_derivative(const T &p[2]) { derivs_.push_back(p); }

    static  const string type_name(int n = -1);
    virtual void io (Piostream& str);

  protected:
    //! support data (node data is elsewhere)

    //! Cubic Hermitian only needs additonal derivatives stored at each node
    //! in the topology.
    vector<T[2]>          derivs_; 
  };

  template <class T>
    const TypeDescription* get_type_description(QuadBicubicHmt<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(QuadBicubicHmt<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    QuadBicubicHmt<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("QuadBicubicHmt");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }


  const int QUADBICUBICHMT_VERSION = 1;
  template <class T>
    void
    QuadBicubicHmt<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), QUADBICUBICHMT_VERSION);
      Pio(stream, derivs_);
      stream.end_class();
    }


} //namespace SCIRun

#endif // QuadBicubicHmt_h
