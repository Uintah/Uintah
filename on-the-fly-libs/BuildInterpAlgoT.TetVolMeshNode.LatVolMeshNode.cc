// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/LatVolField.h"
#include "/usr/local/SCIRun/src/Core/Datatypes/TetVolMesh.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Fields/BuildInterpolant.h"
using namespace SCIRun;

extern "C" {
BuildInterpAlgo* maker() {
  return scinew BuildInterpAlgoT<TetVolMesh, TetVolMesh::Node, LatVolMesh, LatVolMesh::Node, LatVolField<vector<pair<TetVolMesh::Node::index_type, double> > > >;
}
}
