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

    inline void calc_clipped_se(const BBox& obj_bbox, const BBox& bbox,
				const Vector& diag, int nx, int ny, int nz,
				int &sx, int &sy, int &sz,
				int &ex, int &ey, int &ez);

    // All other methods are the same as in parent grid class

private:

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

inline void HierarchicalGrid::calc_clipped_se(const BBox& obj_bbox, const BBox& bbox,
					      const Vector& diag,
					      int nx, int ny, int nz,
					      int& sx, int& sy, int& sz,
					      int& ex, int& ey, int& ez)
{
  Vector s((obj_bbox.min()-bbox.min())/diag);
  Vector e((obj_bbox.max()-bbox.min())/diag);
  sx=(int)(s.x()*nx);
  sy=(int)(s.y()*ny);
  sz=(int)(s.z()*nz);
  ex=(int)(e.x()*nx);
  ey=(int)(e.y()*ny);
  ez=(int)(e.z()*nz);
  sx=Max(Min(sx, nx-1), 0);
  sy=Max(Min(sy, ny-1), 0);
  sz=Max(Min(sz, nz-1), 0);
  ex=Max(Min(ex, nx-1), 0);
  ey=Max(Min(ey, ny-1), 0);
  ez=Max(Min(ez, nz-1), 0);
}

} // end namespace rtrt

#endif
