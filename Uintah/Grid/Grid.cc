
#include "Grid.h"
#include "Level.h"

Grid::Grid()
{
}

Grid::~Grid()
{
}

int Grid::numLevels() const
{
    return levels.size();
}

LevelP& Grid::getLevel(int l)
{
    return levels[l];
}

void Grid::addLevel(const LevelP& level)
{
    levels.push_back(level);
}
