
#include <Core/Datatypes/MeshBase.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID MeshBase::type_id(type_name(), "MeshBaseData", NULL);


MeshBase::~MeshBase() 
{
}

const double MESHBASE_VERSION = 1.0;

void 
MeshBase::io(Piostream& stream) {

  stream.begin_class(type_name().c_str(), MESHBASE_VERSION);
  stream.end_class();
}

const string 
MeshBase::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "MeshBase";
  return name;
}


}
