#ifndef SCIRUN_CORE_CONTAINERS_BOX_RANGE_QUERIER
#define SCIRUN_CORE_CONTAINERS_BOX_RANGE_QUERIER

#include <Core/Containers/RangeTree.h>
#include <Packages/Uintah/testprograms/TestBoxGrouper/Box.h>
#include <list>
#include <vector>

namespace Uintah {
using namespace SCIRun;
using namespace std;

// Just does a simple linear query for testing only.
// Maybe change it to a range tree when testing performance.
class BoxRangeQuerier
{
public:
  typedef list<const Box*> ResultContainer;
  
  template <class BoxPIterator>
  BoxRangeQuerier(BoxPIterator begin, BoxPIterator end);
  ~BoxRangeQuerier();

  void query(const IntVector& low, const IntVector& high,
	     list<const Box*>&);
  void queryNeighbors(const IntVector& low, const IntVector& high,
		      list<const Box*>&);
private:
  class BoxPoint
  {
  public:
    BoxPoint()
      : d_box(NULL) { }

    BoxPoint(IntVector centerTimes2)
      : d_box(NULL), d_centerTimes2(centerTimes2) { }

    BoxPoint(const BoxPoint& copy)
      : d_box(copy.d_box), d_centerTimes2(copy.d_centerTimes2) {}

    void setBox(const Box* box)
    {
      d_box = box;
      d_centerTimes2 = box->getLow() + box->getHigh();
    }
        
    int operator[](int i) const
    { return d_centerTimes2[i]; }

    const Box* getBox() const
    { return d_box; }
  private:
    const Box* d_box;
    
    // center of the patch multiplied by 2
    IntVector d_centerTimes2;
  };

  RangeTree<BoxPoint, int>* d_rangeTree;  
  IntVector d_maxBoxDimensions;

  // BoxPoint's vector is kept here mostly for memory management
  vector<BoxPoint> d_boxPoints;
};

} // end namespace Uintah

#endif
