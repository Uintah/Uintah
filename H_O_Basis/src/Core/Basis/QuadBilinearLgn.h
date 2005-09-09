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
//    File   : QuadBilinearLgn.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Dec 04 2004

#if !defined(QuadBilinearLgn_h)
#define QuadBilinearLgn_h

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

  //! Class for creating geometrical approximations of Quad meshes
  class QuadApprox {  
  public:
    static double UnitVertices[4][2]; //!< Parametric coordinates of vertices of unit edge
    static int UnitEdges[4][2]; //!< References to vertices of unit edge
    static int UnitFaces[1][4]; //!< References to vertices of unit face

    QuadApprox() {}
    virtual ~QuadApprox() {}
   
    //! Approximate edge for element by piecewise linear segments
    //! return: coords gives parametric coordinates of the approximation.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_edge(const unsigned edge, 
			     const unsigned div_per_unit, 
			     vector<vector<double> > &coords) const
    {
      coords.resize(div_per_unit + 1);
      
      const double p1x = UnitVertices[UnitEdges[edge][0]][0];
      const double p1y = UnitVertices[UnitEdges[edge][0]][1];
      const double dx = UnitVertices[UnitEdges[edge][1]][0] - p1x;
      const double dy = UnitVertices[UnitEdges[edge][1]][1] - p1y;
      
      for(unsigned i = 0; i <= div_per_unit; i ++) {
	const double d = (double)div_per_unit / (double)i;
	vector<double> &tmp = coords[i];
	tmp[0] = p1x + d * dx;
	tmp[1] = p1y + d * dy;
      } 	      
    }
  
    //! return number of vertices per face 
    virtual int get_approx_face_elements() const { return 4; }
  
    //! Approximate faces for element by piecewise linear elements
    //! return: coords gives parametric coordinates at the approximation point.
    //! Use interpolate with coordinates to get the world coordinates.
    virtual void approx_face(const unsigned /* face */, 
			     const unsigned div_per_unit, 
			     vector<vector<vector<double> > > &coords) const
    {
      coords.resize(div_per_unit);
      double d = 1. / div_per_unit;
      for(unsigned j = 0; j < div_per_unit; j++) {
	coords[j].resize(2 * (div_per_unit + 1));
	unsigned e = 0;
	for(unsigned i = 0; i <= div_per_unit; i++) {
	  vector<double> &tmp = coords[j][e++];
	  tmp[0] = (double)div_per_unit / (double)i;
	  tmp[1] = (double)div_per_unit / (double)j;
	  tmp = coords[j][e++];
	  // tmp[0] = (double)div_per_unit/(double)i;
	  tmp[1] = (double)div_per_unit / (double)j + d;
	}
      }
    }
  };
 
  //! Class for searching of parametric coordinates related to a value in Quad meshes and fields
  //! to do

  //! Class with weights and coordinates for 2nd order Gaussian integration
   template <class T>
    class QuadGaussian2 
    {
    public:
      static int GaussianNum;
      static T GaussianPoints[4][2];
      static T GaussianWeights[4];
    };

  template <class T>
    int QuadGaussian2<T>::GaussianNum = 4;

  template <class T>
    T QuadGaussian2<T>::GaussianPoints[4][2] = {
    {0.211324865405, 0.211324865405},
    {0.788675134595, 0.211324865405},
    {0.788675134595, 0.788675134595},
    {0.211324865405, 0.788675134595}};

  template <class T>
    T QuadGaussian2<T>::GaussianWeights[4] = {.25, .25, .25, .25};

  //! Class with weights and coordinates for 3rd order Gaussian integration
  template <class T>
    class QuadGaussian3 
    {
    public:
      static int GaussianNum;
      static T GaussianPoints[9][2];
      static T GaussianWeights[9];
    };

  template <class T>
    int QuadGaussian3<T>::GaussianNum = 9;

  template <class T>
    T QuadGaussian3<T>::GaussianPoints[9][2] = {
    {0.11270166537950, 0.11270166537950}, {0.5, 0.11270166537950}, {0.88729833462050, 0.11270166537950},
    {0.11270166537950, 0.5}, {0.5, 0.5}, {0.88729833462050, 0.5},
    {0.11270166537950, 0.88729833462050}, {0.5, 0.88729833462050}, {0.88729833462050, 0.88729833462050}
  };
  
  template <class T>
    T QuadGaussian3<T>::GaussianWeights[9] = {
    0.07716049378395,   0.12345679007654,   0.07716049378395,
    0.12345679007654,   0.19753086415802,   0.12345679007654,
    0.07716049378395,   0.12345679007654,   0.07716049378395
  };

  //! Class for handling of element of type quad with bilinear lagrangian interpolation
  template <class T>
    class QuadBilinearLgn : public QuadApprox, public QuadGaussian2<double> 
  {
  public:
    typedef T value_type;

    QuadBilinearLgn() {}
    virtual ~QuadBilinearLgn() {}
  
    int polynomial_order() const { return 1; }

    //! get value at parametric coordinate
    template <class ElemData>
      T interpolate(const vector<double> &coords, const ElemData &cd) const
      {
	const double x = coords[0], y = coords[1];  
	return (T)((-1 + x) * (-1 + y) * cd.node0()
		   -(x * (-1 + y)) * cd.node1()
		   +x * y * cd.node2()
		   -((-1 + x) * y) * cd.node3());
      }
    
    //! get first derivative at parametric coordinate
    template <class ElemData>
      void derivate(const vector<double> &coords, const ElemData &cd, 
		    vector<double> &derivs) const
      {
	const double x = coords[0], y = coords[1];  

	derivs.resize(2);

	derivs[0] = (-1 + y) * cd.node0()
	  +(1 - y) * cd.node1()
	  +y * cd.node2()
	  -y * cd.node3();

	derivs[1] = (-1 + x) * cd.node0()
	  -x * cd.node1()
	  +x * cd.node2()
	  +(1 - x) * cd.node3();
      }  

    //! get parametric coordinate for value within the element
    template <class ElemData>
      void get_coords(vector<double> &coords, const T& value, 
		      const ElemData &cd) const
      {
      }  

    static  const string type_name(int n = -1);
    virtual void io (Piostream& str);
  };

  template <class T>
    const TypeDescription* get_type_description(QuadBilinearLgn<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(QuadBilinearLgn<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

  template <class T>
    const string
    QuadBilinearLgn<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("QuadBilinearLgn");
	  return nm;
	} else {
	return find_type_name((T *)0);
      }
    }



  const int QUADBILINEARLGN_VERSION = 1;
  template <class T>
    void
    QuadBilinearLgn<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), QUADBILINEARLGN_VERSION);
      stream.end_class();
    }

} //namespace SCIRun

#endif // QuadBilinearLgn_h
