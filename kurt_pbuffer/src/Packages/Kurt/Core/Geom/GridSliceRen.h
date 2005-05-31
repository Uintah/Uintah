#ifndef GRID_SLICE_REN_H
#define GRID_SLICE_REN_H

#include <Packages/Kurt/Core/Geom/GridVolRen.h>
#include <Packages/Kurt/Core/Geom/BrickGrid.h>
#include <Core/Geometry/Polygon.h>
#include <vector>

namespace Kurt {
using std::vector;
using SCIRun::Polygon;



class GridSliceRen : public GridVolRen
{
public:

  GridSliceRen();
  ~GridSliceRen(){}

  virtual void draw(const BrickGrid& bg, int slices = 0);
  virtual void drawWireFrame(const BrickGrid& bg);

  void draw(Brick& b, Polygon* poly);
  void draw(Polygon* poly);

private:


};

} // end namespace Kurt
#endif
