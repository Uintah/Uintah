#include <Packages/Uintah/testprograms/TestBoxGrouper/Box.h>

using namespace Uintah;

bool Box::isInside(IntVector low, IntVector high) const
{
  IntVector lowOverlap = Max(low, low_);
  IntVector highOverlap = Min(high, high_);
  return (highOverlap.x() >= lowOverlap.x() &&
	  highOverlap.y() >= lowOverlap.y() &&
	  highOverlap.z() >= lowOverlap.z());
}

bool Box::isNeighboring(IntVector low, IntVector high) const
{
  IntVector lowOverlap = Max(low, low_);
  IntVector highOverlap = Min(high, high_);

  int neighboringSides = 0;
  int overlappingSides = 0;
  for (int i = 0; i < 3; i++) {
    if (highOverlap[i] >= lowOverlap[i])
      overlappingSides++;
    if (highOverlap[i] + 1 == lowOverlap[i])
      neighboringSides++;
  }

  // Note: no corner neighbors allowed -- must have two overlapping sides
  return neighboringSides == 1 && overlappingSides == 2;
}

