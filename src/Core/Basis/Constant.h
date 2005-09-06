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
//    File   : Constant.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Mon Aug 23 09:57:14 2004

#if !defined(Constant_h)
#define Constant_h

#include <vector>
#include <string>
#include <Core/Geometry/Point.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Datatypes/Datatype.h>

namespace SCIRun {

  using std::vector;
  using std::string;


  //! used for creation of fields without storage for field variables.
  template <class T>
    class ConstantBasis : public Datatype
  {
  public:
    ConstantBasis() {}
    virtual ~ConstantBasis() {}
  
    unsigned polynomial_order() const { return 0; }

    //! get value at parametric coordinate 
    template <class CellData>
      T interpolate(const vector<double> &coords, const CellData &cd) const
      {
	return cd.elem();
      }
  
    //! get first derivative at parametric coordinate
    template <class CellData>
      T derivate(const vector<double> &coords, const CellData &cd) const
      {
	return (T)0;
      }

    //! get second derivative at parametric coordinate
    template <class CellData>
      T derivate2(const vector<double> &coords, const CellData &cd) const
      {
	return (T)0;
      }

    //! get parametric coordinate for value within the element
    //! iterative solution...
    template <class CellData>
      void get_coords(vector<double> &coords, const T& value, 
		      const CellData &cd) const
      {
	coords.resize(3,0);
      }
    virtual int get_approx_face_elements() const { return 0; }
    static  const string type_name(int n = -1);
    virtual void io (Piostream& str);

  };

  template <class T>
    const string
    ConstantBasis<T>::type_name(int n)
    {
      ASSERT((n >= -1) && n <= 1);
      if (n == -1)
	{
	  static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
	  return name;
	}
      else if (n == 0)
	{
	  static const string nm("ConstantBasis");
	  return nm;
	}
      else 
	{
	  return find_type_name((T *)0);
	}
    }

  template <class T>
    const TypeDescription* get_type_description(ConstantBasis<T> *)
    {
      static TypeDescription* td = 0;
      if(!td){
	const TypeDescription *sub = SCIRun::get_type_description((T*)0);
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
	(*subs)[0] = sub;
	td = scinew TypeDescription(ConstantBasis<T>::type_name(0), subs, 
				    string(__FILE__),
				    "SCIRun");
      }
      return td;
    }

#define CONSTANTBASIS_VERSION 1

  template <class T>
    void
    ConstantBasis<T>::io(Piostream &stream)
    {
      stream.begin_class(type_name(-1), CONSTANTBASIS_VERSION);
      stream.end_class();
    }

}
#endif // Constant_h
