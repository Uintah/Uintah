#ifndef __CELL_H__
#define __CELL_H__

#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>

#include <vector>

namespace Uintah {

using namespace SCIRun;

class Cell {
public:
  std::vector<particleIndex> particles;
  
  void  insert(const particleIndex& p);
private:
};

} // End namespace Uintah

#endif //__CELL_H__

