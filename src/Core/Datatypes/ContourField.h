
#ifndef Datatypes_ContourField_h
#define Datatypes_ContourField_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ContourMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>
#include <vector>

namespace SCIRun {

using std::string;

template <class Data>
class ContourField: public GenericField< ContourMesh, vector<Data> > { 

public:

  ContourField() :
    GenericField<ContourMesh, vector<Data> >() {}
  ContourField(Field::data_location data_at) :
    GenericField<ContourMesh, vector<Data> >(data_at) {}
  ContourField(ContourMeshHandle mesh, Field::data_location data_at) : 
    GenericField<ContourMesh, vector<Data> >(mesh, data_at) {}
  
  virtual ~ContourField(){}

  virtual ContourField<Data> *clone() const 
    { return new ContourField<Data>(*this); }
 
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
private:
  static Persistent* maker();
};

#define CONTOURFIELD_VERSION 1

template <class Data>
Persistent* 
ContourField<Data>::maker()
{
  return scinew ContourField<Data>;
}

template <class Data>
PersistentTypeID
ContourField<Data>::type_id(type_name(),
		GenericField<ContourMesh, vector<Data> >::type_name(),
                maker); 

template <class Data>
void
ContourField<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name().c_str(), CONTOURFIELD_VERSION);
  GenericField<ContourMesh, vector<Data> >::io(stream);
  stream.end_class();                                                         
}


template <class Data>
const string
ContourField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "ContourField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

} // end namespace SCIRun

#endif // Datatypes_ContourField_h
















