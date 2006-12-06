#include <Package/Uintah/testprograms/TestBoxGrouper/Box.h>
#include <list>

namespace Uintah {
using namespace SCIRun;
using namespace std;

// Just does a simple linear query for testing only.
// Maybe change it to a range tree when testing performance.
class BoxRangeQuerier
{
public:
  typedef list<Box*> ResultContainer;
  
  BoxRangeQuerier(const vector<Box*>& boxes)
    : boxes_(boxes) {}

  list<Box*> query(const IntVector& low, const IntVector& high);
  list<Box*> queryNeighbors(const IntVector& low, const IntVector& high);
private:
  vector<Box*> boxes_;
};

} // end namespace Uintah
