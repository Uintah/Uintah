#ifndef SCI_Wangxl_Datatypes_Mesh_DEdge_h
#define SCI_Wangxl_Datatypes_Mesh_DEdge_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/Triple.h>

namespace Wangxl {

using namespace SCIRun;

class DCell;

typedef triple<DCell*, int, int> DEdge;

}

#endif
