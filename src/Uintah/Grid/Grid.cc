/* REFERENCED */
static char *id="@(#) $Id$";

#include "Grid.h"
#include "Level.h"

namespace Uintah {
namespace Grid {

Grid::Grid()
{
}

Grid::~Grid()
{
}

int Grid::numLevels() const
{
    return d_levels.size();
}

LevelP& Grid::getLevel( int l )
{
    return d_levels[ l ];
}

void Grid::addLevel(const LevelP& level)
{
    d_levels.push_back( level );
}

} // end namespace Grid
} // end namespace Uintah


