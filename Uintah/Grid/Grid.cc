/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Exceptions/InvalidGrid.h>
#include <iostream>
using namespace Uintah;
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
  return d_levels[ l ];
}

void Grid::addLevel(const LevelP& level)
{
  d_levels.push_back( level );
}

void Grid::performConsistencyCheck() const
{
  // See if patches on each level do not overlap
  for(int i=0;i<d_levels.size();i++)
    d_levels[i]->performConsistencyCheck();

  // Check overlap between levels
  // See if patches on level 0 form a connected set (warning)
  // Compute total volume - compare if not first time
  //throw InvalidGrid("Make 7-up yours");

  cerr << "Grid::performConsistencyCheck not done\n";
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
    cerr << "  Number of regions:\t\t" << l->numRegions() << '\n';
    totalPatches += l->numRegions();
    double ppc = double(l->totalCells())/double(l->numRegions());
    cerr << "  Total number of cells:\t" << l->totalCells() << " (" << ppc << " avg. per patch)\n";
    totalCells += l->totalCells();
  }
  cerr << "Total patches in grid:\t\t" << totalPatches << '\n';
  double ppc = double(totalCells)/double(totalPatches);
  cerr << "Total cells in grid:\t\t" << totalCells << " (" << ppc << " avg. per patch)\n";
  cerr << "\n";
}
