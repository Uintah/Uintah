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
//    File   : CrvQuadraticLgn.h
//    Author : Marty Cole, Frank B. Sachse
//    Date   : Nov 30 2004

#if !defined(CrvQuadraticLgn_h)
#define CrvQuadraticLgn_h

#include <CrvLinearLgn.h>

namespace SCIRun {

  template <class T>
    class CrvQuadraticLgn : public CrvApprox
  {
  public:
    static int GaussianNum;
    static double GaussianPoints[2][1];
    static double GaussianWeights[2];

    CrvQuadraticLgn() {}
    virtual ~CrvQuadraticLgn() {}
  
    // Value at coord
    template <class CellData>
      T interpolate(const vector<double> &coords, const CellData &cd) const
      {
	const double x=coords[0];  
	return T((1 - 3*x + 2*x*x) * cd.node0() 
		 -4*(-1 + x)*x* cd.node1()
		 +x*(-1 + 2*x)* nodes_[cd.edge0_index()]);
      }
  
    //! First derivative at coord.
    template <class CellData>
      void derivate(const vector<double> &coords, const CellData &cd, 
		    vector<double> &derivs) const
      {
	const double x=coords[0];  
    
	derivs.size(1);

	derivs[0] = (-3 + 4*x) * cd.node0() 
	  +(4 - 8*x)* cd.node1()
	  +(-1 + 4*x)* nodes_[cd.edge0_index()];
      }
  
    //! add a node value corresponding to edge
    void add_node_value(const T &p) { nodes_.push_back(p); }

    static  const string type_name(int n = -1);

    //! return the parametric coordinates for value within the element.
    template <class CellData>
      void get_coords(vector<double> &coords, const T& value, 
		      const CellData &cd) const  
      {
	CrvLocate< CrvQuadraticLgn<T> > CL;
	CL.get_coords(this, coords, value, cd);
      };
     
    virtual void io (Piostream& str);
  protected:
    //! Additional support values.

    //! Quadratic Lagrangian only needs additional nodes stored for each edge
    //! in the topology.
    vector<T>          nodes_; 
  };


  template <class T>
    const TypeDescription* get_type_description(CrvQuadraticLgn<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(CrvQuadraticLgn<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    CrvQuadraticLgn<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("CrvQuadraticLgn");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }

  const int CRVQUADRATICLGN_VERSION = 1;
  template <class T>
    void
    CrvQuadraticLgn<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), CRVQUADRATICLGN_VERSION);
      Pio(stream, nodes_);
      stream.end_class();
    }
  
  template <class T>
    int CrvQuadraticLgn<T>::GaussianNum = 2;

  template <class T>
    double CrvQuadraticLgn<T>::GaussianPoints[2][1] = {{0.211324865405}, {0.788675134595}};

  template <class T>
    double CrvQuadraticLgn<T>::GaussianWeights[2] = {.5, .5};


} //namespace SCIRun

#endif // CrvQuadraticLgn_h
