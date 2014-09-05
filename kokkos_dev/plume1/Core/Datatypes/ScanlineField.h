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



#ifndef Datatypes_ScanlineField_h
#define Datatypes_ScanlineField_h

#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;


template <class Data>
class ScanlineField : public GenericField< ScanlineMesh, vector<Data> >
{
public:
  ScanlineField();
  ScanlineField(int order);
  ScanlineField(ScanlineMeshHandle mesh, int order);
  virtual ScanlineField<Data> *clone() const;
  virtual ~ScanlineField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent* maker();
};



template <class Data>
ScanlineField<Data>::ScanlineField()
  : GenericField<ScanlineMesh, vector<Data> >()
{
}


template <class Data>
ScanlineField<Data>::ScanlineField(int order)
  : GenericField<ScanlineMesh, vector<Data> >(order)
{
}


template <class Data>
ScanlineField<Data>::ScanlineField(ScanlineMeshHandle mesh,
				   int order)
  : GenericField<ScanlineMesh, vector<Data> >(mesh, order)
{
}


template <class Data>
ScanlineField<Data> *
ScanlineField<Data>::clone() const
{
  return new ScanlineField(*this);
}
  

template <class Data>
ScanlineField<Data>::~ScanlineField()
{
}


#define SCANLINE_FIELD_VERSION 1

template <class Data>
Persistent* 
ScanlineField<Data>::maker()
{
  return scinew ScanlineField<Data>;
}

template <class Data>
PersistentTypeID
ScanlineField<Data>::type_id(type_name(-1),
		GenericField<ScanlineMesh, vector<Data> >::type_name(-1),
                maker); 

template <class Data>
void
ScanlineField<Data>::io(Piostream &stream)
{
  /*int version=*/stream.begin_class(type_name(-1), SCANLINE_FIELD_VERSION);
  GenericField<ScanlineMesh, vector<Data> >::io(stream);
  stream.end_class();                                                         
}


template <class Data>
const string
ScanlineField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "ScanlineField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
ScanlineField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if (n == -1) {
    static TypeDescription* tdn1 = 0;
    if (tdn1 == 0) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      tdn1 = scinew TypeDescription(name, subs, path, namesp);
    } 
    td = tdn1;
  }
  else if(n == 0) {
    static TypeDescription* tdn0 = 0;
    if (tdn0 == 0) {
      tdn0 = scinew TypeDescription(name, 0, path, namesp);
    }
    td = tdn0;
  }
  else {
    static TypeDescription* tdnn = 0;
    if (tdnn == 0) {
      tdnn = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
    td = tdnn;
  }
  return td;
}

} // end namespace SCIRun

#endif // Datatypes_ScanlineField_h
