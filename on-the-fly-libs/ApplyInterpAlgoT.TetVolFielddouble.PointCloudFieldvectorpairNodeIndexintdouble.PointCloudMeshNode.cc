// This is an autamatically generated file, do not edit!
#include <utility>
#include <vector>
#include "/usr/local/SCIRun/src/Core/Datatypes/FieldIndex.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolField.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/ApplyInterpolant.h"
using namespace SCIRun;
using namespace std;

extern "C" {
ApplyInterpAlgo* maker() {
  return scinew ApplyInterpAlgoT<TetVolField<double> , PointCloudField<vector<pair<NodeIndex<int> ,double> > > , PointCloudMesh::Node, PointCloudField<double> >;
}
}
