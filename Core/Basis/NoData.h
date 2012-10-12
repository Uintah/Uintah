/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
//  
//    File   : NoData.h
//    Author : Martin Cole, Frank B. Sachse
//    Date   : Mon Aug 23 09:57:14 2004

#if !defined(NoData_h)
#define NoData_h

#include <Core/Basis/Basis.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/TypeName.h>

namespace SCIRun {

//! Class for describing unit geometry of NoDataBasis 
class NoDataUnitElement {
public: 
  NoDataUnitElement() {}
  virtual ~NoDataUnitElement() {}

  static double unit_vertices[1][1]; //!< Parametric coordinates of vertices 
  static int unit_edges[1][1];    //!< References to vertices of unit edge 

  //! return dimension of domain 
  static int domain_dimension() { return -1; } 
  //! return number of vertices
  static int number_of_vertices() { return 0; }
  //! return number of vertices in mesh
  static int number_of_mesh_vertices() { return 0; }
  //! return degrees of freedom
  static int dofs() { return 0; } 

  //! return number of edges 
  static int number_of_edges() { return 0; } 
  //! return number of vertices per face 
  static int vertices_of_face() { return 0; } 
  //! return number of faces per cell 
  static int faces_of_cell() { return 0; } 

  static double volume() { return 0.; } //!< return volume
};


//! Class for handling of element without storage for field variables
template <class T> //! for compilation consistency
  class NoDataBasis : public BasisSimple<T>, public NoDataUnitElement
{
public:
  typedef T value_type;

  NoDataBasis() {}
  virtual ~NoDataBasis() {}
  
  static int polynomial_order() { return -1; }
  
  static  const std::string type_name(int n = -1);
  virtual void io (Piostream& str); 
};

template <class T>
const std::string
NoDataBasis<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const std::string nm("NoDataBasis");
    return nm;
  }
  else 
  {
    return find_type_name((T *)0);
  }
}

template <class T>
const TypeDescription* get_type_description(NoDataBasis<T> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("NoDataBasis", subs, 
				std::string(__FILE__), 
				"SCIRun", 
				TypeDescription::BASIS_E);
  }
  return td;
}

#define NODATABASIS_VERSION 1

template <class T>
void
NoDataBasis<T>::io(Piostream &stream)
{
  stream.begin_class(get_type_description(this)->get_name(),
                     NODATABASIS_VERSION);
  stream.end_class();
}

} // end namespace SCIRun

#endif // NoData_h
