// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadraticTetVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/ManageFieldData.h"
using namespace SCIRun;

extern "C" {
ManageFieldDataAlgoMesh* maker() {
  return scinew ManageFieldDataAlgoMeshScalar<QuadraticTetVolField<double> >;
}
}
