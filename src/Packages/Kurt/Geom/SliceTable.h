#ifndef SLICETABLE_H
#define SLICETABLE_H


#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Ray.h>
#include <SCICore/Geometry/Vector.h>
#include "Brick.h"

namespace Kurt {
namespace GeomSpace  {


using namespace SCICore::Geometry;
using namespace SCICore::Containers;


class SliceTable 
{
public:

  SliceTable(Point min, Point max, Ray view, int levels, int slices);

  ~SliceTable();

  void getParameters(const Brick*,double& tmin, double& tmax, double& dt) const;

private:
  int maxLevels;
  Ray view;
  int slices;
  int order[8];
  double minT, maxT, DT;
};


}  // namespace GeomSpace
} // namespace Kurt
#endif
