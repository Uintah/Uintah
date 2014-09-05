// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/TriSurfField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/ManageFieldData.h"
using namespace SCIRun;

extern "C" {
ManageFieldDataAlgoMesh* maker() {
  return scinew ManageFieldDataAlgoMeshScalar<TriSurfField<double> >;
}
}
