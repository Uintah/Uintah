
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Exceptions/InvalidGrid.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Geometry/BBox.h>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

Grid::Grid()
{
}

Grid::~Grid()
{
}

int Grid::numLevels() const
{
  return (int)d_levels.size();
}

const LevelP& Grid::getLevel( int l ) const
{
  ASSERTRANGE(l, 0, numLevels());
  return d_levels[ l ];
}

Level* Grid::addLevel(const Point& anchor, const Vector& dcell, int id)
{
  Level* level = scinew Level(this, anchor, dcell, (int)d_levels.size(), id);  
  d_levels.push_back( level );
  return level;
}

void Grid::performConsistencyCheck() const
{
  // Verify that patches on a single level do not overlap
  for(int i=0;i<(int)d_levels.size();i++)
    d_levels[i]->performConsistencyCheck();

  // Check overlap between levels
  // See if patches on level 0 form a connected set (warning)
  // Compute total volume - compare if not first time

  //cerr << "Grid::performConsistencyCheck not done\n";
}

void Grid::printStatistics() const
{
  cerr << "Grid statistics:\n";
  cerr << "Number of levels:\t\t" << numLevels() << '\n';
  unsigned long totalCells = 0;
  unsigned long totalPatches = 0;
  for(int i=0;i<numLevels();i++){
    LevelP l = getLevel(i);
    cerr << "Level " << i << ":\n";
    if (l->getPeriodicBoundaries() != IntVector(0,0,0))
      cerr << "  Periodic boundaries:\t\t" << l->getPeriodicBoundaries()
	   << '\n';
    cerr << "  Number of patches:\t\t" << l->numPatches() << '\n';
    totalPatches += l->numPatches();
    double ppc = double(l->totalCells())/double(l->numPatches());
    cerr << "  Total number of cells:\t" << l->totalCells() << " (" << ppc << " avg. per patch)\n";
    totalCells += l->totalCells();
  }
  cerr << "Total patches in grid:\t\t" << totalPatches << '\n';
  double ppc = double(totalCells)/double(totalPatches);
  cerr << "Total cells in grid:\t\t" << totalCells << " (" << ppc << " avg. per patch)\n";
  cerr << "\n";
}

//////////
// Computes the physical boundaries for the grid
void Grid::getSpatialRange(BBox& b) const
{
  // just call the same function for all the levels
  for(int l=0; l < numLevels(); l++) {
    getLevel(l)->getSpatialRange(b);
  }
}





