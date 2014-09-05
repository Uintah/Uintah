// This is an autamatically generated file, do not edit!
#include <utility>
#include <vector>
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldIndex.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/InterpolantToTransferMatrix.h"
using namespace SCIRun;
using namespace std;

extern "C" {
Interp2TransferAlgo* maker() {
  return scinew Interp2TransferAlgoT<LatVolField<vector<pair<NodeIndex<int> ,double> > > , LatVolMesh::Node>;
}
}
