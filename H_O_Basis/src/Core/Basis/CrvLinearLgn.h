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
//    File   : CrvLinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 03 2004

#if !defined(CrvLinearLgn_h)
#define CrvLinearLgn_h

#include <vector>
#include <string>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Transform.h>
#include <float.h>

namespace SCIRun {

  using std::vector;
  using std::string;

  //! Class for creating geometrical approximations of Crv meshes
  class CrvApprox {
  public:
    static double UnitVertices[1][2]; //!< Parametric coordinates of vertices of unit edge
    static int UnitEdges[1][2]; //!< References to vertices of unit edge

    CrvApprox() {}
    virtual ~CrvApprox() {}
  
    //! Approximate edge for element by piecewise linear segments
    //! return: coords gives parametric coordinates of the approximation.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_edge(const unsigned /* edge */, 
			     const unsigned div_per_unit, 
			     vector<vector<double> > &coords) const
    {
      coords.resize(div_per_unit + 1);
      for(unsigned i = 0; i <= div_per_unit; i++) {
	vector<double> &tmp = coords[i];
	tmp.resize(1);
	tmp[0] = (double)div_per_unit / (double)i;
      }
    }
  
    //! return number of vertices per face 
    virtual int get_approx_face_elements() const { return 0; }
  
    //! Approximate faces for element by piecewise linear elements
    //! return: coords gives parametric coordinates at the approximation point.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_face(const unsigned /* face */, 
			     const unsigned /* div_per_unit */, 
			     vector<vector<double> > &coords) const
    {
      coords.resize(0);
    }
  };
 

  //! Class for searching of parametric coordinates related to a value in Crv meshes and fields
  template <class CrvBasis>
    class CrvLocate {
  public:

    typedef typename CrvBasis::value_type T;

    CrvLocate() {}
    virtual ~CrvLocate() {}
 
    //! iterative solution...
    template <class CellData>
      void get_coords(CrvBasis *pCrv, vector<double> &coords, const T& value, 
			      const CellData &cd) const  
      {     
	//! Step 1: get a good guess on the domain, evaluate equally spaced points 
	//!         on the domain and use the closest as our starting point for 
	//!         Newton iteration.
      
	vector<double>  cur, last;
	initial_guess(pCrv, value, cd, cur);
     
	//! Now cur has our initialization param for Newton iteration.
	//! Step 2: Newton iteration.
	//! f(u) =C'(u) dot (C(u) - P)
	//! the distance from P to C(u) is minimum when f(u) = 0     
	do {
	  last = cur;
	  T dif = pCrv->interpolate(cur, cd)-value; 
	  T grad = pCrv->derivate(cur, cd);
	  cur+=dif/grad;
	} while (sqrt((last-cur)*(last-cur)) > 0.00001);
	coords.clear();
	coords.push_back(cur);
      };
 
 protected:
    //! find a reasonable initial guess for starting Newton iteration.
    template <class CellData>
      void initial_guess(CrvBasis *pCrv, const T &val, const CellData &cd, vector<double> cur) const
      {
	double dist = HUGE;
	
	int end = 4;
	vector<double> guess(1,0.L);
	for (int x = 0; x <= end; x++) {
	  guess[0] = x / (double) end;
	  T dv = pCrv->interpolate(guess, cd)-val;
	  T cur_d = sqrt(dv*dv);
	  if (cur_d < dist) {
	    dist = cur_d;
	    cur = guess;
	  }
	}
      }
  };


  //! Class for handling of element of type curve with linear lagrangian interpolation
  template <class T>
    class CrvLinearLgn : public CrvApprox
  {
  public:
    typedef T value_type;

    static int    GaussianNum;
    static double GaussianPoints[1][1];
    static double GaussianWeights[1];

    int polynomial_order() const { return 1; }

    CrvLinearLgn() : CrvApprox() {}
      virtual ~CrvLinearLgn() {}
  
      virtual void approx_edge(const unsigned /* edge */, 
			       const unsigned /*div_per_unit*/,
			       vector<vector<double> > &coords) const
      {
	coords.resize(2);
	vector<double> &tmp = coords[0];
	tmp.resize(1);
	tmp[0] = 0.0;
	tmp = coords[1];
	tmp.resize(1);
	tmp[0] = 1.0;
      }

      //! get value at parametric coordinate
      template <class CellData>
	T interpolate(const vector<double> &coords, const CellData &cd) const
	{
	  const double x=coords[0];  
	  return (T)((1-x)*cd.node0()+x*cd.node1());
	}
    
      //! get first derivative at parametric coordinate
      template <class CellData>
	void derivate(const vector<double> &coords, const CellData &cd, 
		      vector<double> &derivs) const
	{
	  derivs.size(1);

	  derivs[0] = -cd.node0() + cd.node1();
	}

      //! get parametric coordinate for value within the element
      template <class CellData>
	void get_coords(vector<double> &coords, const T& value, 
			const CellData &cd) const  
	{
	  double denominator = cd.node1() - cd.node0();
	  double numerator   = value - cd.node0();
	  coords[0] = numerator / denominator;
	}
 
      static  const string type_name(int n = -1);

      virtual void io (Piostream& str);
  protected:
  };

  template <class T>
    const TypeDescription* get_type_description(CrvLinearLgn<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(CrvLinearLgn<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    CrvLinearLgn<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("CrvLinearLgn");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }

  const int CRVLINEARLGN_VERSION = 1;
  template <class T>
    void
    CrvLinearLgn<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), CRVLINEARLGN_VERSION);
      stream.end_class();
    }

  template <class T>
    int CrvLinearLgn<T>::GaussianNum = 1;

  template <class T>
    double CrvLinearLgn<T>::GaussianPoints[1][1] = {{0.5}};

  template <class T>
    double CrvLinearLgn<T>::GaussianWeights[1] = {1.};

} //namespace SCIRun

#endif // CrvLinearLgn_h
