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



#ifndef Datatypes_PointCloudField_h
#define Datatypes_PointCloudField_h

#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

template <class Data>
class PointCloudField: public GenericField< PointCloudMesh, vector<Data> >
{ 
public:

  PointCloudField();
  PointCloudField(int order);
  PointCloudField(PointCloudMeshHandle mesh, int order);  
  virtual PointCloudField<Data> *clone() const; 
  virtual ~PointCloudField();

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent* maker();
};

const int POINT_CLOUD_FIELD_VERSION = 1;

template <class Data>
Persistent* 
PointCloudField<Data>::maker()
{
  return scinew PointCloudField<Data>;
}

template <class Data>
PersistentTypeID
PointCloudField<Data>::type_id(type_name(-1),
		GenericField<PointCloudMesh, vector<Data> >::type_name(-1),
                maker); 

template <class Data>
void
PointCloudField<Data>::io(Piostream &stream)
{
  /*int version=*/stream.begin_class(type_name(-1), POINT_CLOUD_FIELD_VERSION);
  GenericField<PointCloudMesh, vector<Data> >::io(stream);
  stream.end_class();                                                         
}

template <class Data>
PointCloudField<Data>::PointCloudField()
  :  GenericField<PointCloudMesh, vector<Data> >()
{
}


template <class Data>
PointCloudField<Data>::PointCloudField(int order)
  : GenericField<PointCloudMesh, vector<Data> >(order)
{
}


template <class Data>
PointCloudField<Data>::PointCloudField(PointCloudMeshHandle mesh,
				       int order)
  : GenericField<PointCloudMesh, vector<Data> >(mesh, order)
{
}
  

template <class Data>
PointCloudField<Data>::~PointCloudField()
{
}


template <class Data>
PointCloudField<Data> *
PointCloudField<Data>::clone() const 
{
  return new PointCloudField<Data>(*this);
}

 
template <class Data>
const string
PointCloudField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "PointCloudField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class T> 
const TypeDescription*
PointCloudField<T>::get_type_description(int n) const
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

#endif // Datatypes_PointCloudField_h
