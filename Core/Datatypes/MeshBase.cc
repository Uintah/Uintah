
#include <Core/Datatypes/MeshBase.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID MeshBase::type_id(type_name(0), "MeshBaseData", NULL);


MeshBase::~MeshBase() 
{
}

const double MESHBASE_VERSION = 1.0;

void 
MeshBase::io(Piostream& stream) {

  stream.begin_class(MeshBase::type_name(0).c_str(), MESHBASE_VERSION);
  stream.end_class();
}

const string 
MeshBase::type_name(int)
{
  static const string name = "MeshBase";
  return name;
}

}
