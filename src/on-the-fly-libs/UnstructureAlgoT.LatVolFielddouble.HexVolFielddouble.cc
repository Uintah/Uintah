// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/Unstructure.h"
using namespace SCIRun;

extern "C" {
UnstructureAlgo* maker() {
  return scinew UnstructureAlgoT<LatVolField<double> ,HexVolField<double> >;
}
}
