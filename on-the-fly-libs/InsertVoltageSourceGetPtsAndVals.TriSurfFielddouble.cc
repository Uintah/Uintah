// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/TriSurfField.h"
#include "/usr/local/SCIRun/src/Packages/BioPSE/Dataflow/Modules/Forward/InsertVoltageSource.h"
using namespace SCIRun;

extern "C" {
InsertVoltageSourceGetPtsAndValsBase* maker() {
  return scinew InsertVoltageSourceGetPtsAndVals<TriSurfField<double> , TriSurfMesh::Node>;
}
}
