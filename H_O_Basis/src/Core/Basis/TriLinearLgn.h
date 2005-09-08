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
//    File   : TriLinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(TriLinearLgn_h)
#define TriLinearLgn_h

#include <vector>
#include <string>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/Transform.h>
#include <float.h>

namespace SCIRun {

  using std::vector;
  using std::string;
  
  //! Class for creating geometrical approximations of Tri meshes
  class TriApprox {  
  public:
    static double UnitVertices[3][2]; //!< Parametric coordinates of vertices of unit edge
    static int UnitEdges[3][2]; //!< References to vertices of unit edge
    static int UnitFaces[1][3]; //!< References to vertices of unit face

    TriApprox() {}
    virtual ~TriApprox() {}
  
    //! Approximate edge for element by piecewise linear segments
    //! return: coords gives parametric coordinates of the approximation.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_edge(const unsigned edge, const unsigned div_per_unit, 
			     vector<vector<double> > &coords) const
    {
      coords.resize(div_per_unit+1);

      const double *v0 = UnitVertices[UnitEdges[edge][0]];
      const double *v1 = UnitVertices[UnitEdges[edge][1]];

      const double &p1x = v0[0];
      const double &p1y = v0[1];
      const double dx = v1[0] - p1x;
      const double dy = v1[1] - p1y;

      for(unsigned i = 0; i <= div_per_unit; i++) {
	const double d = (double)div_per_unit / (double)i;
	vector<double> &tmp = coords[i];
	tmp[0] = p1x + d * dx;
	tmp[1] = p1y + d * dy;
      } 	
    } 
  
    //! return number of vertices per face 
    virtual int get_approx_face_elements() const { return 3; }
  
    //! Approximate faces for element by piecewise linear elements
    //! return: coords gives parametric coordinates at the approximation point.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_face(const unsigned /* face */, 
			     const unsigned div_per_unit, 
			     vector<vector<vector<double> > > &coords) const
    {
      coords.resize(div_per_unit);
      const double d = 1. / div_per_unit;

      for(unsigned j = 0; j < div_per_unit; j++) {
	const double dj = (double)div_per_unit / (double)j;
	unsigned e = 0;
	coords[j].resize((div_per_unit - j) * 2 + 1);
	vector<double> &tmp = coords[j][e++];
	tmp[0] = 0;
	tmp[1] = dj;
	for(unsigned i = 0; i<div_per_unit - j; i++) {
	  const double di = (double)div_per_unit / (double)i;
	  tmp = coords[j][e++];
	  tmp[0] = di;
	  tmp[1] = dj + d;
	  tmp = coords[j][e++];
	  tmp[0] = di + d;
	  tmp[1] = dj;
	}
      }
    }
  };

  //! Class for searching of parametric coordinates related to a value in Tri meshes and fields
  //! to do

  //! Class for handling of element of type triangle with linear lagrangian interpolation
  template <class T>
    class TriLinearLgn : public TriApprox {
  public:
    static int GaussianNum;
    static double GaussianPoints[3][2];
    static double GaussianWeights[3];

    TriLinearLgn() {}
    virtual ~TriLinearLgn() {}
  
    virtual int polynomial_order() const { return 1; }

    virtual void approx_edge(const unsigned edge, 
			     const unsigned /* div_per_unit */,
			     vector<vector<double> > &coords) const
    {
      coords.resize(2);
      vector<double> &tmp = coords[0];
      tmp[0] = UnitVertices[UnitEdges[edge][0]][0];
      tmp[1] = UnitVertices[UnitEdges[edge][0]][1];
      tmp = coords[1];
      tmp[0] = UnitVertices[UnitEdges[edge][1]][0];
      tmp[1] = UnitVertices[UnitEdges[edge][1]][1];
    }

    virtual void approx_face(const unsigned /* face */, 
			     const unsigned div_per_unit,
			     vector<vector<vector<double> > > &coords) const
    {
      coords.resize(1);
      coords[0].resize(3);
      vector<double> &tmp = coords[0][0];
      tmp[0] = 0;
      tmp[1] = 0;
      tmp = coords[0][1];
      tmp[0] = 1;
      tmp[1] = 0;
      tmp = coords[0][2];
      tmp[0] = 0;
      tmp[1] = 1;
    }

    //! get value at parametric coordinate
    template <class ElemData>
      T interpolate(const vector<double> &coords, const ElemData &cd) const
      {
	const double x = coords[0], y = coords[1];  
	return (T)((1 - x - y) * cd.node0() + x * cd.node1() + y * cd.node2());
      }
  
    //! get first derivative at parametric coordinate
    template <class ElemData>
      void derivate(const vector<double> &coords, const ElemData &cd, 
		    vector<double> &derivs) const
      {
	derivs.resize(2);

	derivs[0] = -cd.node0() + cd.node1();
	derivs[1] = -cd.node0() + cd.node2();
      }
	
    //! get the parametric coordinate for value within the element
    //! iterative solution...
    template <class ElemData>
      void get_coords(vector<double> &coords, const T& value, 
		      const ElemData &cd) const;  

    static const string type_name(int n = -1);

    virtual void io (Piostream& str);

  protected:

    double distance(const T&, const T&) const;
  };


  template <class T>
    const TypeDescription* get_type_description(TriLinearLgn<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(TriLinearLgn<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    TriLinearLgn<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("TriLinearLgn");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }


  template <class T>
    template <class ElemData>
    void 
    TriLinearLgn<T>::get_coords(vector<double> &coords, const T& value, 
				const ElemData &cd) const
    {
    }

  const int TRILINEARLGN_VERSION = 1;
  template <class T>
    void 
    TriLinearLgn<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), TRILINEARLGN_VERSION);
      stream.end_class();
    }

  template <class T>
    int TriLinearLgn<T>::GaussianNum = 3;

  template <class T>
    double TriLinearLgn<T>::GaussianPoints[3][2] = {
    {1./6.,1./6.}, {2./3.,1./6.}, {1./6.,2./3.}};

  template <class T>
    double TriLinearLgn<T>::GaussianWeights[3] = {1./3., 1./3., 1./3.};

} //namespace SCIRun

#endif // TriLinearLgn_h
