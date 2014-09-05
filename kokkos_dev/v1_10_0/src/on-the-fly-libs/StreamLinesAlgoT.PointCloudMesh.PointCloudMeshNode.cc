// This is an autamatically generated file, do not edit!
#include "/usr/local/SCIRun/src/Core/Datatypes/PointCloudMesh.h"
#include "/usr/local/SCIRun/src/Dataflow/Modules/Visualization/StreamLines.h"
using namespace SCIRun;

extern "C" {
StreamLinesAlgo* maker() {
  return scinew StreamLinesAlgoT<PointCloudMesh, PointCloudMesh::Node>;
}
}
