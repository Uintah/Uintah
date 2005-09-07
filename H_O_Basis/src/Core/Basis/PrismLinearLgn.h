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
//    File   : PrismLinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(PrismLinearLgn_h)
#define PrismLinearLgn_h

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

  class PrismApprox {  
  public:
    static double UnitVertices[6][3]; //! Parametric coordinates of vertices of unit edge
    static int UnitEdges[9][3]; //! References to vertices of unit edge
    static int UnitFaces[5][4]; //! References to vertices of unit face

    PrismApprox() {}
    virtual ~PrismApprox() {}
  
    //! Approximate edge for element by piecewise linear segments
    //! return: coords gives parametric coordinates of the approximation.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_edge(const unsigned edge, const unsigned div_per_unit, 
			     vector<vector<double> > &coords) const
    {
      coords.resize(div_per_unit + 1);

      const double *v0 = UnitVertices[UnitEdges[edge][0]];
      const double *v1 = UnitVertices[UnitEdges[edge][1]];

      const double &p1x = v0[0];
      const double &p1y = v0[1];
      const double &p1z = v0[2];
      const double dx = v1[0] - p1x;
      const double dy = v1[1] - p1y;
      const double dz = v1[2] - p1z;

      for(unsigned i = 0; i <= div_per_unit; i++) {
	const double d = (double)div_per_unit / (double)i;
	vector<double> &tmp = coords[i];
	tmp[0] = p1x + d * dx;
	tmp[1] = p1y + d * dy;
	tmp[2] = p1z + d * dz;
      } 	
    }
  
    //! return number of vertices per face 
    virtual int get_approx_face_elements() const  { return 3; }
  
    //! Approximate faces for element by piecewise linear elements
    //! return: coords gives parametric coordinates at the approximation point.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_face(const unsigned face, const unsigned div_per_unit, 
			     vector<vector<vector<double> > > &coords) const
    {	
      int fe = (UnitFaces[face][4] != -1 ? 2 : 1);
      coords.resize(fe * div_per_unit);
	
      for(int f = 0; f<2; f++) {
	double *v0, *v1, *v2;

	if (f==0) {
	  v0 = UnitVertices[UnitFaces[face][0]];
	  v1 = UnitVertices[UnitFaces[face][1]];
	  v2 = UnitVertices[UnitFaces[face][3]];
	} else {
	  v0 = UnitVertices[UnitFaces[face][2]];
	  v1 = UnitVertices[UnitFaces[face][3]];
	  v2 = UnitVertices[UnitFaces[face][1]];
	}

	const double d = 1. / div_per_unit;
	for(unsigned j = 0; j < div_per_unit; j++) {
	  const double dj = (double)div_per_unit / (double)j;
	  unsigned e = 0;
	  coords[j + f * div_per_unit].resize((div_per_unit - j) * 2 + 1);
	  vector<double> &tmp = coords[j + f * div_per_unit][e++];
	  tmp[0] = v0[0] + dj * (v2[0] - v0[0]);
	  tmp[1] = v0[1] + dj * (v2[1] - v0[1]);
	  tmp[2] = v0[2] + dj * (v2[2] - v0[2]);
	  for(unsigned i = 0; i < div_per_unit - j; i++) {
	    const double di = (double)div_per_unit / (double)i;
	    tmp = coords[j + f * div_per_unit][e++];
	    tmp[0] = v0[0] + (dj + d) * (v2[0] - v0[0]) + di * (v1[0] - v0[0]);
	    tmp[1] = v0[1] + (dj + d) * (v2[1] - v0[1]) + di * (v1[1] - v0[1]);
	    tmp[2] = v0[2] + (dj + d) * (v2[2] - v0[2]) + di * (v1[2] - v0[2]);
	    tmp = coords[j + f * div_per_unit][e++];
	    tmp[0] = v0[0] + dj * (v2[0] - v0[0]) + (di + d) * (v1[0] - v0[0]);
	    tmp[1] = v0[1] + dj * (v2[1] - v0[1]) + (di + d) * (v1[1] - v0[1]);
	    tmp[2] = v0[2] + dj * (v2[2] - v0[2]) + (di + d) * (v1[2] - v0[2]);
	  }
	}
      }
    }
  };


  template <class T>
    class PrismLinearLgn : public PrismApprox {
  public:
    typedef T value_type;

    static int GaussianNum;
    static double GaussianPoints[6][2];
    static double GaussianWeights[6];
  
    PrismLinearLgn() {}
    virtual ~PrismLinearLgn() {}

    int polynomial_order() const { return 1; }
  
    //! get value at parametric coordinate 
    template <class ElemData>
      T interpolate(const vector<double> &coords, const ElemData &cd) const
      {
	const double x = coords[0], y = coords[1], z = coords[2];  
 
	return (T)((-1 + x + y) * (-1 + z) * cd.node0()
		   - (x * (-1 + z)) * cd.node1()
		   - (y * (-1 + z)) * cd.node2()
		   - ((-1 + x + y) * z) * cd.node3()
		   +x * z * cd.node4()
		   +y * z * cd.node5());
      }
  
    //! get first derivative at parametric coordinate
    template <class ElemData>
      void derivate(const vector<double> &coords, const ElemData &cd, 
		    vector<double> &derivs) const
      {
	const double x = coords[0], y = coords[1], z = coords[2]; 

	derivs.resize(3);
 
	derivs[0] = (-1 + z) * cd.node0()
	  +(1 - z) * cd.node1()
	  - z * cd.node3()
	  +z * cd.node4();
      
	derivs[1] = (-1 + z) * cd.node0()
	  + (1 - z) * cd.node2()
	  - z * cd.node3()
	  + z * cd.node5();
      
	derivs[2] = (-1 + x + y) * cd.node0()
	  - x * cd.node1()
	  - y * cd.node2()
	  + (1 - x - y) * cd.node3()
	  + x * cd.node4()
	  + y * cd.node5();
      }

    //! get parametric coordinate for value within the element
    //! iterative solution...
    template <class ElemData>
      void get_coords(vector<double> &coords, const T& value, 
		      const ElemData &cd) const;  

    static  const string type_name(int n = -1);

    virtual void io (Piostream& str);

  protected:
    double distance(const T&, const T&) const;
  };


  template <class T>
    const TypeDescription* get_type_description(PrismLinearLgn<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(PrismLinearLgn<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    PrismLinearLgn<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("PrismLinearLgn");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }


  const int PRISMLINEARLGN_VERSION = 1;
  template <class T>
    void
    PrismLinearLgn<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), PRISMLINEARLGN_VERSION);
      stream.end_class();
    }

  template <class T>
    int PrismLinearLgn<T>::GaussianNum = 6;

  template <class T>
    double PrismLinearLgn<T>::GaussianPoints[6][2] = {
    {1./6.,1./6., 0.211324865405}, {2./3.,1./6., 0.211324865405}, 
    {1./6.,2./3., 0.211324865405}, {1./6.,1./6., 0.788675134595}, 
    {2./3.,1./6., 0.788675134595}, {1./6.,2./3., 0.788675134595}};

  template <class T>
    double PrismLinearLgn<T>::GaussianWeights[6] = 
    {1./6., 1./6., 1./6., 1./6., 1./6., 1./6.};
  
} //namespace SCIRun

#endif // PrismLinearLgn_h
