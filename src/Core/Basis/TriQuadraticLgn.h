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
//    File   : TriQuadraticLgn.h
//    Author : Martin Cole, Frank Sachse
//    Date   : Dec 04 2004

#if !defined(TriQuadraticLgn_h)
#define TriQuadraticLgn_h

#include <TriLinearLgn.h>

namespace SCIRun {

  //! Class for handling of element of type triangle with linear quadratic interpolation
  template <class T>
    class TriQuadraticLgn : public TriApprox<T> {
  public:
    typedef T value_type;
    
    static int GaussianNum;
    double GaussianPoints[7][2];
    double GaussianWeights[7];
   
    TriQuadraticLgn() {}
    virtual ~TriQuadraticLgn() {}

    //! get value at parametric coordinate
    template <class ElemData>
      T interpolate(const vector<double> &coords, const ElemData &cd) const
      {
	const double x=coords[0], y=coords[1];  
	return (1 + 2*x*x - 3*y + 2*y*y + x*(-3 + 4*y))*cd.node0()
	  +x*(-1 + 2*x)*cd.node1()
	  +y*(-1+ 2*y)*cd.node2()
	  -4*x*(-1 + x + y)*nodes_[cd.edge0_index()]
	  +4*x*y*nodes_[cd.edge1_index()]
	  -4*y*(-1 + x + y)*nodes_[cd.edge2_index()]
	  };
  
    //! get first derivative at parametric coordinate
    template <class ElemData>
      void derivate(const vector<double> &coords, const ElemData &cd, 
		    vector<double> &derivs) const
      {
	const double x=coords[0], y=coords[1];  

	derivs.size(2);

	derivs[0]=(-3 + 4*x + 4*y)*cd.node0()
	  +(-1 + 4*x)*cd.node1()
	  -4*(-1 + 2*x + y)*nodes_[cd.edge0_index()]
	  +4*y*nodes_[cd.edge1_index()]
	  -4*y*nodes_[cd.edge2_index()];

	derivs[1]=(-3 + 4*x + 4*y)*cd.node0()
	  +(-1 +4*y)*cd.node2()
	  -4*x*nodes_[cd.edge0_index()]
	  +4*x*nodes_[cd.edge1_index()]
	  -4*(-1 + x +2*y)*nodes_[cd.edge2_index()];
      };

    //! get the parametric coordinate for value within the element
    //! iterative solution...
    template <class ElemData>
      void get_coords(vector<double> &coords, const T& value, 
		      const ElemData &cd) const;  

    //! add a node value corresponding to edge
    void add_node_value(const T &p) { nodes_.push_back(p); }

    static const string type_name(int n = -1);

    virtual void io (Piostream& str);

  protected:

    double distance(const T&, const T&) const;

    //! Additional support values.

    //! Quadratic Lagrangian only needs additional nodes stored for each edge
    //! in the topology.
    vector<T>          nodes_; 
  };


  template <class T>
    const TypeDescription* get_type_description(TriQuadraticLgn<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(TriQuadraticLgn<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    TriQuadraticLgn<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("TriQuadraticLgn");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }


  template <class T>
    template <class ElemData>
    void 
    TriQuadraticLgn<T>::get_coords(vector<double> &coords, const T& value, 
				   const ElemData &cd) const
    {
    }

  const int TRIQUADRATICLGN_VERSION = 1;
  template <class T>
    void
    TriQuadraticLgn<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), TRIQUADRATICLGN_VERSION);
      Pio(stream, nodes_);
      stream.end_class();
    }

  template <class T>
    int TriQuadraticLgn<T>::GaussianNum = 7;

  template <class T>
    double TriQuadraticLgn<T>::GaussianPoints[7][2] = {
    {0.1012865073, 0.1012865073}, {0.7974269853, 0.1012865073}, {0.1012865073, 0.7974269853},
    {0.4701420641, 0.0597158717}, {0.4701420641, 0.4701420641}, {0.0597158717, 0.4701420641},
    {0.3333333333, 0.3333333333}};

  template <class T>
    double TriQuadraticLgn<T>::GaussianWeights[7] = 
    {0.1259391805, 0.1259391805, 0.1259391805, 0.1323941527, 0.1323941527, 0.1323941527, 0.0225};

} //namespace SCIRun

#endif // TriQuadraticLgn_h
