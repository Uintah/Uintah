// This is an autamatically generated file, do not edit!
#include <utility>
#include <vector>
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldIndex.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/QuadSurfField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/HexVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/ApplyInterpolant.h"
using namespace SCIRun;
using namespace std;

extern "C" {
ApplyInterpAlgo* maker() {
  return scinew ApplyInterpAlgoT<HexVolField<double> , QuadSurfField<vector<pair<NodeIndex<int> ,double> > > , QuadSurfMesh::Node, QuadSurfField<double> >;
}
}
