#include <Package/Uintah/testprograms/TestBoxGrouper/Box.h>
#include <Package/Uintah/testprograms/TestBoxGrouper/BoxRangeQuerier.h>

list<Box*> BoxRangeQuerier::query(const IntVector& low, const IntVector& high)
{
  list<Box*> results;
  for (int i = 0; i < boxes_; i++) {
    if (boxes_[i]->inside(low, high))
      results.push_back(boxes_[i]);
  }
  return results;
}

list<Box*> BoxRangeQuerier::queryNeighbors(const IntVector& low, const IntVector& high)
{
  list<Box*> results;
  for (int i = 0; i < boxes_; i++) {
    if (boxes_[i]->isNeighboring(low, high))
      results.push_back(boxes_[i]);
  }
  return results;
}
