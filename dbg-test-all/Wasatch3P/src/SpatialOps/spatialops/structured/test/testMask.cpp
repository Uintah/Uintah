#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/Nebo.h>
#include <test/TestHelper.h>
#include <spatialops/structured/FieldHelper.h>
#include <spatialops/structured/ExternalAllocators.h>

#ifdef ENABLE_THREADS
#include <spatialops/ThreadPool.h>
#include <boost/thread/barrier.hpp>
#endif

#include <limits>
#include <cmath>
#include <list>

#include <sstream>
#include <fstream>

#include <spatialops/structured/SpatialMask.h>
#include <spatialops/NeboMask.h>

using namespace SpatialOps;
using std::cout;
using std::endl;

int main(int argc, const char *argv[])
{
  const bool print = false;
  TestHelper status(print);

  typedef SVolField FieldT;

  const int nghost = 1;
  const GhostData ghost(nghost);
  const BoundaryCellInfo bcinfo = BoundaryCellInfo::build<FieldT>(false,false,false);
  MemoryWindow w(IntVec(10, 10, 10));
  FieldT f(w, bcinfo, ghost, NULL, InternalStorage);
  FieldT result(w, bcinfo, ghost, NULL, InternalStorage);
  FieldT::iterator ir = result.begin();

  result <<= 4;

  //result field + nghost for proper indexing
  *(ir + w.flat_index(IntVec(3, 3, 0) + nghost)) = 3;
  *(ir + w.flat_index(IntVec(4, 3, 0) + nghost)) = 3;
  *(ir + w.flat_index(IntVec(3, 4, 0) + nghost)) = 3;
  *(ir + w.flat_index(IntVec(4, 4, 0) + nghost)) = 3;

  *(ir + w.flat_index(IntVec(1, 1, 0) + nghost)) = 7;
  *(ir + w.flat_index(IntVec(6, 1, 0) + nghost)) = 7;
  *(ir + w.flat_index(IntVec(1, 6, 0) + nghost)) = 7;
  *(ir + w.flat_index(IntVec(6, 6, 0) + nghost)) = 7;

  std::vector<IntVec> maskSet;
  std::vector<IntVec> maskSet2;

  //mask of center
  maskSet.push_back(IntVec(3, 3, 0));
  maskSet.push_back(IntVec(4, 3, 0));
  maskSet.push_back(IntVec(3, 4, 0));
  maskSet.push_back(IntVec(4, 4, 0));

  //mask of square corners
  maskSet2.push_back(IntVec(1, 1, 0));
  maskSet2.push_back(IntVec(6, 1, 0));
  maskSet2.push_back(IntVec(1, 6, 0));
  maskSet2.push_back(IntVec(6, 6, 0));

  //SpatialMasks
  SpatialMask<FieldT> mask(f, maskSet);
  SpatialMask<FieldT> mask2(f, maskSet2);

  SpatialMask<FieldT> typedef SpatialMaskT;
  NeboMask<Initial, FieldT> typedef NeboMaskT;
  f <<= cond(mask, 3)
            (mask2, 7)
            (4);

  status( (display_fields_compare(result, f, print, print)), "Cond version");

  f <<= 4;
  masked_assign(mask, f, 3);
  masked_assign(mask2, f, 7);

  status( (display_fields_compare(result, f, print, print)), "masked_assign version");

  if( status.ok() ) {
    std::cout << "ALL TESTS PASSED :)" << std::endl;
    return 0;
  } else {
    std::cout << "******************************" << std::endl
              << " At least one test FAILED! :(" << std::endl
              << "******************************" << std::endl;
    return -1;
  }
}
