#ifndef HIERARCHICALGRID_H
#define HIERARCHICALGRID_H 1

#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Thread/WorkQueue.h>
#include <Packages/rtrt/Core/Group.h>

namespace rtrt {

using SCIRun::WorkQueue;

struct GridTree;

class HierarchicalGrid : public Grid {

public:

    //
    // Create a hierarchical grid of maximum 3 levels deep
    //
    // nside -- number of grid cells at level1
    // nSideLevel2 -- number of grid cells at level2
    // nSideLevel3 -- number of grid cells at level3
    // minObjects1 -- number of objects per cell before they are embedded in 
    //               another grid at level1
    // minObjects2 -- number of objects per cell before they are embedded in 
    //               another grid at level2
    //
    
    HierarchicalGrid( Object* obj, 
		      int nside,
		      int nSideLevel2, int nSideLevel3,
		      int minObjects1, int minObjects2, int np );

    HierarchicalGrid( Object* obj, 
		      int nside,
		      int nSideLevel2, int nSideLevel3,
		      int minObjects1, int minObjects2,
		      const BBox &b, int level );

    virtual ~HierarchicalGrid( void );

    virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);

    void calc_clipped_se(const BBox& obj_bbox, const BBox& bbox,
				const Vector& diag, int nx, int ny, int nz,
				int &sx, int &sy, int &sz,
				int &ex, int &ey, int &ez);

    // All other methods are the same as in parent grid class

private:

  int total_;

  typedef struct {
    double dx;
    double dy;
    double dz;
    double maxradius;
    int pp_offset;
    int scratchsize;
    Array1<int> whichCell;
    Array1<int> whichCellPos;
    Array1<Group*> objList;
  } PData;
  PData pdata;
  WorkQueue* work;
  void gridit(int);
  int _nSidesLevel2, _nSidesLevel3;
  int _minObjects1, _minObjects2;
  int _level;
  int np;
  static int L1Cells, L2Cells;
  static int L1CellsWithChildren, L2CellsWithChildren;
  static int LeafCells, TotalLeafPrims;
};

} // end namespace rtrt

#endif
