/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  QuadSurfField.h
 *
 *  Written by:
 *   Michael Callahan
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_QuadSurfField_h
#define Datatypes_QuadSurfField_h

#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>


namespace SCIRun {

template <class T> 
class QuadSurfField : public GenericField<QuadSurfMesh, vector<T> >
{
public:
  QuadSurfField();
  QuadSurfField(int order);
  QuadSurfField(QuadSurfMeshHandle mesh, int order);
  virtual QuadSurfField<T> *clone() const;
  virtual ~QuadSurfField();
  
  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent *maker();
};

// Pio defs.
const int QUAD_SURF_FIELD_VERSION = 1;

template <class T>
Persistent *
QuadSurfField<T>::maker()
{
  return scinew QuadSurfField<T>;
}

template <class T>
PersistentTypeID 
QuadSurfField<T>::type_id(type_name(-1), 
		    GenericField<QuadSurfMesh, vector<T> >::type_name(-1),
		    maker);


template <class T>
void 
QuadSurfField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), QUAD_SURF_FIELD_VERSION);
  GenericField<QuadSurfMesh, vector<T> >::io(stream);
  stream.end_class();
}

template <class T>
QuadSurfField<T>::QuadSurfField()
  : GenericField<QuadSurfMesh, vector<T> >()
{
}

template <class T>
QuadSurfField<T>::QuadSurfField(int order) :
  GenericField<QuadSurfMesh, vector<T> >(order)
{
}

template <class T>
QuadSurfField<T>::QuadSurfField(QuadSurfMeshHandle mesh, int order)
  : GenericField<QuadSurfMesh, vector<T> >(mesh, order)
{
} 

template <class T>
QuadSurfField<T> *
QuadSurfField<T>::clone() const
{
  return new QuadSurfField(*this);
}

template <class T>
QuadSurfField<T>::~QuadSurfField()
{
}

template <class T> 
const string 
QuadSurfField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "QuadSurfField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
QuadSurfField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if(!td){
    if (n == -1) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      td = scinew TypeDescription(name, subs, path, namesp);
    }
    else if(n == 0) {
      td = scinew TypeDescription(name, 0, path, namesp);
    }
    else {
      td = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
  }
  return td;
}

} // end namespace SCIRun

#endif // Datatypes_QuadSurfField_h



















