#ifndef SCI_Wangxl_Datatypes_Mesh_VMEdge_h
#define SCI_Wangxl_Datatypes_Mesh_VMEdge_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/Triple.h>

namespace Wangxl {

using namespace SCIRun;

class VMCell;

typedef triple<VMCell*, int, int> VMEdge;

}

#endif
