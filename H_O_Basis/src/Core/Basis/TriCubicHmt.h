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
//    File   : TriCubicHmt.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(TriCubicHmt_h)
#define TriCubicHmt_h

#include <TriLinearLgn.h>

namespace SCIRun {

  //! Triangular topology
  template <class T>
    class TriCubicHmt : public TriApprox<T> {
  public:
    static int GaussianNum;
    double GaussianPoints[7][2];
    double GaussianWeights[7];
    
    TriCubicHmt() {}
    virtual ~TriCubicHmt() {}
    
    int polynomial_order() const { return 3; }

    //! get value at parametric coordinate
    template <class ElemData>
      T interpolate(const vector<double> &coords, const ElemData &cd) const
      {
	const double x=coords[0], y=coords[1];  
	return (-1 + x + y)*(-1 - x + 2*x*x - y + 11*x*y + 2*y*y)*cd.node0()
	  +x*(-1+ x + y)*(-1 + x + 2*y)*derivs_[cd.node0_index()][0]
	  +y*(-1 + x + y)*(-1 + 2*x + y)*derivs_[cd.node0_index()][1]
	  -(x*(-3*x +2*x*x + 7*y - 7*x*y - 7*y*y))*cd.node1()
	  +x*(-x + x*x + 2*y - 2*x*y- 2*y*y)*derivs_[cd.node1_index()][0]
	  +x*y*(-1 + 2*x + y)*derivs_[cd.node1_index()][1]
	  -(y*(7*x - 7*x*x - 3*y - 7*x*y+ 2*y*y))*cd.node2()
	  +x*y*(-1 + x + 2*y)*derivs_[cd.node2_index()][0]
	  +y*(2*x - 2*x*x - y - 2*x*y + y*y)*derivs_[cd.node2_index()][1]
	  -27*x*y*(-1 + x + y)*nodes_[cd.elem];    
      }
  
    //! get first derivative at parametric coordinate
    template <class ElemData>
      void derivate(const vector<double> &coords, const ElemData &cd, 
		    vector<double> &derivs) const
      {
	const double x=coords[0], y=coords[1];  

	derivs.size(2);

	derivs[0]=(6*x*x + 13*(-1 + y)*y + x*(-6 + 26*y))*cd.node0()
	  +(1 + 3*x*x - 3*y + 2*y*y + x*(-4 + 6*y))*derivs_[cd.node0_index()][0]
	  +y*(-3 + 4*x + 3*y)*derivs_[cd.node0_index()][1]
	  +(-6*x*x + 7*(-1 + y)*y + 2*x*(3 + 7*y))*cd.node1()
	  +(3*x*x - 2*(-1 + y)*y - 2*x*(1 + 2*y))*derivs_[cd.node1_index()][0]
	  +y*(-1 + 4*x + y)*derivs_[cd.node1_index()][1]
	  +7*y*(-1 + 2*x + y)*cd.node2()
	  +y*(-1 + 2*x + 2*y)*derivs_[cd.node2_index()][0]
	  -2*y*(-1 + 2*x + y)*derivs_[cd.node2_index()][1]
	  -27*y*(-1 + 2*x + y)*nodes_[cd.elem];
    
	derivs[1]=(13*x*x + 6*(-1+ y)*y + 13*x*(-1 + 2*y))*cd.node0()
	  +x*(-3 + 3*x + 4*y)*derivs_[cd.node0_index()][0]
	  +(1 + 2*x*x - 4*y + 3*y*y + x*(-3 + 6*y))*derivs_[cd.node0_index()][1]
	  +7*x*(-1 + x + 2*y)*cd.node1()
	  -2*x*(-1 + x + 2*y)*derivs_[cd.node1_index()][0]
	  +x*(-1 + 2*x + 2*y)*derivs_[cd.node1_index()][1]
	  +(7*x*x - 6*(-1 + y)*y + 7*x*(-1 + 2*y))*cd.node2()
	  +x*(-1 + x + 4*y)*derivs_[cd.node2_index()][0]
	  +(-2*x*x + x*(2 - 4*y) + y*(-2 + 3*y))*derivs_[cd.node2_index()][1]
	  -27*x*(-1 + x + 2*y)*nodes_[cd.elem];
      }
  
    //! get the parametric coordinate for value within the element.
    //! iterative solution...
    template <class ElemData>
      void get_coords(vector<double> &coords, const T& value, 
		      const ElemData &cd) const;  

    //! add derivative values (dx, dy) for nodes.
    void add_derivative(const T &p[2]) { derivs_.push_back(p); }

    //! add a node value. (must correspond to center of triangle)
    void add_node(const T &p) { nodes_.push_back(p); }

    static const string type_name(int n = -1);

    virtual void io (Piostream& str);

  protected:

    double distance(const T&, const T&) const;

    //! Additional support values.

    //! Cubic Hermitian needs additonal derivatives stored at each node
    //! in the topology.
    vector<T[2]>          derivs_; 

    //! Cubic Lagrangian needs additional nodes stored for each edge
    //! in the topology.
    vector<T>          nodes_; 
  };


  template <class T>
    const TypeDescription* get_type_description(TriCubicHmt<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(TriCubicHmt<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    TriCubicHmt<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("TriCubicHmt");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }


  template <class T>
    template <class ElemData>
    void 
    TriCubicHmt<T>::get_coords(vector<double> &coords, const T& value, 
			       const ElemData &cd) const
    {
    }

  const int TRICUBICHMT_VERSION = 1;
  template <class T>
    void
    TriCubicHmt<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), TRICUBICHMT_VERSION);
      Pio(stream, derivs_);
      Pio(stream, nodes_);
      stream.end_class();
    }

  template <class T>
    int TriCubicHmt<T>::GaussianNum = 7;

  template <class T>
    double TriCubicHmt<T>::GaussianPoints[7][2] = {
    {0.1012865073, 0.1012865073}, {0.7974269853, 0.1012865073}, {0.1012865073, 0.7974269853},
    {0.4701420641, 0.0597158717}, {0.4701420641, 0.4701420641}, {0.0597158717, 0.4701420641},
    {0.3333333333, 0.3333333333}};

  template <class T>
    double TriCubicHmt<T>::GaussianWeights[7] = 
    {0.1259391805, 0.1259391805, 0.1259391805, 0.1323941527, 0.1323941527, 0.1323941527, 0.0225};


} //namespace SCIRun

#endif // TriCubicHmt_h
