#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/CCA/Components/Schedulers/SchedulerCommon.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

#include <iostream>
#include <sstream>

using namespace std;
using namespace Uintah;


RegridderCommon::RegridderCommon(const ProcessorGroup* pg) : Regridder(), UintahParallelComponent(pg)
{
  d_filterType = FILTER_STAR;

  cout << "I am contructing a RegridderCommon." << endl;

  CCVariable<int> flaggedCells;
  flaggedCells.rewindow(IntVector(0,0,0), IntVector(11,11,11));
  flaggedCells.initialize(0);
  flaggedCells[IntVector(5,5,5)] = 1;

  CCVariable<int> dilatedFlaggedCells;
  dilatedFlaggedCells.rewindow(IntVector(0,0,0), flaggedCells.size());

  Dilate( flaggedCells, dilatedFlaggedCells, FILTER_STAR, IntVector(1,1,1) );
}

RegridderCommon::~RegridderCommon()
{
  for (int k = 0; k < d_maxLevels; k++) {
    delete d_patchActive[k];
    delete d_patchCreated[k];
    delete d_patchDeleted[k];
  }  

  d_patchActive.clear();
  d_patchCreated.clear();
  d_patchDeleted.clear();
}

bool RegridderCommon::needRecompile(double time, double delt, const GridP& grid)
{
  bool retval = d_newGrid;
  d_newGrid = false;
  return retval;
}

void RegridderCommon::problemSetup(const ProblemSpecP& params, 
                                   const GridP& oldGrid,
				   const SimulationStateP& state)

{
  d_sharedState = state;

  ProblemSpecP regrid_spec = params->findBlock("Regridder");
  if (!regrid_spec)
    throw ProblemSetupException("Must specify a Regridder section for"
                                " AMR problems\n");

  // get max num levels
  regrid_spec->require("max_levels", d_maxLevels);

  // get cell refinement ratio
  // get simple ratio - allow user to just say '2'
  int simpleRatio = -1;
  int size;
  regrid_spec->get("simple_refinement_ratio", simpleRatio);
  regrid_spec->get("cell_refinement_ratio", d_cellRefinementRatio);
  size = (int) d_cellRefinementRatio.size();

  if (simpleRatio != -1 && size > 0)
    throw ProblemSetupException("Cannot specify both simple_refinement_ratio"
                                " and cell_refinement_ratio\n");
  if (simpleRatio == -1 && size == 0)
    throw ProblemSetupException("Must specify either simple_refinement_ratio"
                                " or cell_refinement_ratio\n");

  // as it is not required to have cellRefinementRatio specified to all levels,
  // expand it to all levels here for convenience in looking up later.
  if (simpleRatio != -1) {
    d_cellRefinementRatio.push_back(IntVector(simpleRatio, simpleRatio, simpleRatio));
    size = 1;
  }

  IntVector lastRatio = d_cellRefinementRatio[size - 1];
  if (size < d_maxLevels) {
    d_cellRefinementRatio.resize(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++)
      d_cellRefinementRatio[i] = lastRatio;
  }
  
  // Check for existence TODO

  // get lattice refinement ratio, expand it to max levels
  regrid_spec->require("lattice_refinement_ratio", d_latticeRefinementRatio);
  size = (int) d_latticeRefinementRatio.size();
  lastRatio = d_latticeRefinementRatio[size - 1];
  if (size < d_maxLevels) {
    d_latticeRefinementRatio.resize(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++)
      d_latticeRefinementRatio[i] = lastRatio;
  }

  // get other init parameters
  d_cellCreationDilation = IntVector(1,1,1);
  d_cellDeletionDilation = IntVector(1,1,1);
  d_minBoundaryCells = IntVector(1,1,1);

  regrid_spec->get("cell_creation_dilation", d_cellCreationDilation);
  regrid_spec->get("cell_deletion_dilation", d_cellDeletionDilation);
  regrid_spec->get("min_boundary_cells", d_minBoundaryCells);

  const LevelP level0 = oldGrid->getLevel(0);
  d_cellNum.resize(d_maxLevels);
  d_patchNum.resize(d_maxLevels);
  d_patchSize.resize(d_maxLevels);
  d_patchActive.resize(d_maxLevels);
  d_patchCreated.resize(d_maxLevels);
  d_patchDeleted.resize(d_maxLevels);
  
  // get level0 resolution
  IntVector low, high;
  level0->findCellIndexRange(low, high);
  d_cellNum[0] = high-low;
  const Patch* patch = level0->selectPatchForCellIndex(IntVector(0,0,0));
  d_patchSize[0] = patch->getHighIndex() - patch->getLowIndex();
  d_patchNum[0] = calculateNumberOfPatches(d_cellNum[0], d_patchSize[0]);
  d_patchActive[0] = new CCVariable<int>;
  d_patchCreated[0] = new CCVariable<int>;
  d_patchDeleted[0] = new CCVariable<int>;
  d_patchActive[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchCreated[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchDeleted[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchActive[0]->initialize(1);
  d_patchCreated[0]->initialize(0);
  d_patchDeleted[0]->initialize(0);
  if ( Mod( d_patchSize[0], d_latticeRefinementRatio[0] ) != IntVector(0,0,0) ) {
    throw InternalError("Regridder: patch size is not divisible by lattice ratio on level 0!");
  }

  for (int k = 0; k < d_maxLevels; k++) {
    d_cellNum[k] = d_cellNum[k-1] * d_cellRefinementRatio[k-1];
    d_patchSize[k] = d_patchSize[k-1] * d_cellRefinementRatio[k-1] /
      d_latticeRefinementRatio[k-1];
    d_patchNum[k] = calculateNumberOfPatches(d_cellNum[k], d_patchSize[k]);
    d_patchActive[k] = new CCVariable<int>;
    d_patchCreated[k] = new CCVariable<int>;
    d_patchDeleted[k] = new CCVariable<int>;
    d_patchActive[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchCreated[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchDeleted[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchActive[k]->initialize(0);
    d_patchCreated[k]->initialize(0);
    d_patchDeleted[k]->initialize(0);
    if (k < (d_maxLevels-1)) {
      if ( Mod( d_patchSize[k], d_latticeRefinementRatio[k] ) != IntVector(0,0,0) ) {
	ostringstream msg;
	msg << "Regridder: patch size is not divisible by lattice ratio on level " << k; 
	throw InternalError(msg.str());
      }
    }
  }
}

bool RegridderCommon::flaggedCellsExist(CCVariable<int>& flaggedCells, IntVector low, IntVector high)
{
  if (low > high) {
    throw InternalError("Regridder has given flagCellsExist incorrect parameters!");
  }

  for(CellIterator iter(low, high); !iter.done(); iter++) {
    IntVector idx(*iter);
    if (flaggedCells[idx])
      return true;
  }
  return false;
}

IntVector RegridderCommon::calculateNumberOfPatches(IntVector& cellNum, IntVector& patchSize)
{
  IntVector patchNum = Ceil(Vector(cellNum.x(), cellNum.y(), cellNum.z()) / patchSize);
  IntVector remainder = Mod(cellNum, patchSize);
  IntVector small = And(Less(remainder, patchSize / IntVector(2,2,2)), Greater(remainder,IntVector(0,0,0)));
  
  for ( int i = 0; i < 3; i++ ) {
    if (small[i])
      patchNum[small[i]]--;
  }

  return patchNum;
}


IntVector RegridderCommon::Less    (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() < b.x(), a.y() < b.y(), a.z() < b.z());
}

IntVector RegridderCommon::Greater (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() > b.x(), a.y() > b.y(), a.z() > b.z());
}

IntVector RegridderCommon::And     (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() & b.x(), a.y() & b.y(), a.z() & b.z());
}

IntVector RegridderCommon::Mod     (const IntVector& a, const IntVector& b)
{
  return IntVector(a.x() % b.x(), a.y() % b.y(), a.z() % b.z());
}

IntVector RegridderCommon::Ceil    (const Vector& a)
{
  return IntVector(static_cast<int>(ceil(a.x())), static_cast<int>(ceil(a.y())), static_cast<int>(ceil(a.z())));
}

void RegridderCommon::GetFlaggedCells ( const GridP& oldGrid, int levelIdx, DataWarehouse* dw )
{
  // This needs to be fixed for Parallel cases.

  LevelP level = oldGrid->getLevel(levelIdx);

  IntVector minIdx = (*(level->patchesBegin()))->getCellLowIndex();
  IntVector maxIdx = (*(level->patchesBegin()))->getCellHighIndex();

  // This could be a problem because of extra cells.
  
  for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++ ) {
    const Patch* patch = *patchIter;
    minIdx = Min( minIdx, patch->getCellLowIndex() );
    maxIdx = Max( maxIdx, patch->getCellHighIndex() );
  }
  
  d_flaggedCells[levelIdx] = new CCVariable<int>;
  d_dilatedCellsCreated[levelIdx] = new CCVariable<int>;
  d_dilatedCellsDeleted[levelIdx] = new CCVariable<int>;
  
  d_flaggedCells[levelIdx]->rewindow( minIdx, maxIdx );
  d_dilatedCellsCreated[levelIdx]->rewindow( minIdx, maxIdx );
  d_dilatedCellsDeleted[levelIdx]->rewindow( minIdx, maxIdx );

  // This is only a first step, getting the dilation cells in serial.
  // This is a HUGE memory waste.

  for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++ ) {
    const Patch* patch = *patchIter;
    IntVector l(patch->getCellLowIndex());
    IntVector h(patch->getCellHighIndex());
    
    constCCVariable<int> refineFlag;
    dw->get(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch, Ghost::None, 0);
    
    for(CellIterator iter(l, h); !iter.done(); iter++){
      IntVector idx(*iter);
      if (refineFlag[idx])
	(*d_flaggedCells[levelIdx])[idx] = true;
    }
  }
}

void RegridderCommon::Dilate( CCVariable<int>& flaggedCells, CCVariable<int>& dilatedFlaggedCells, int filterType, IntVector depth )
{
  if ((depth.x() < 0) || (depth.y() < 0) || (depth.z() < 0))  throw InternalError("Regridder given a bad dilation depth!");

  // we'll find a better way to do this
  CCVariable<int> filter;
  filter.rewindow(IntVector(0,0,0), IntVector(2,2,2)*depth+IntVector(2,2,2));
  filter.initialize(0);

  cout << "Size of Filter = " << filter.size() << endl;

  switch (filterType) {

    case FILTER_STAR: {
      for (int x = 0; x < 2*depth.x()+1; x++) {
	for (int y = 0; y < 2*depth.y()+1; y++) {
	  for (int z = 0; z < 2*depth.z()+1; z++) {
	    if ((fabs(static_cast<float>(x - depth.x()))/(depth.x()+1e-16) +
		 fabs(static_cast<float>(y - depth.y()))/(depth.y()+1e-16) +
		 fabs(static_cast<float>(z - depth.z()))/(depth.z()+1e-16)) <= 1.0) {
	      filter[IntVector(x,y,z)] = 1;
	    }
	  }
	}
      }
      break;
    }

    case FILTER_BOX: {
      filter.initialize(1);
      break;
    }

    default: throw InternalError("Regridder given a bad dilation filter type!");
  }

  cout << "----------------------------------------------------------------" << endl;
  cout << "FILTER" << endl;

  for (int z = 0; z < 2*depth.z()+1; z++) {
    for (int y = 0; y < 2*depth.y()+1; y++) {
      for (int x = 0; x < 2*depth.x()+1; x++) {
	cout << filter[IntVector(x,y,z)] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }


  IntVector arraySize = flaggedCells.size();
  IntVector dilatedArraySize = dilatedFlaggedCells.size();

  if (arraySize != dilatedArraySize) {
    throw InternalError("Original flagged cell array and dilated flagged cell array do not possess the same size!");
  }

  IntVector low, high;

  for (int x = 0; x < arraySize.x(); x++) {
    for (int y = 0; y < arraySize.y(); y++) {
      for (int z = 0; z < arraySize.z(); z++) {
	cout << "Dilating (" << x << "," << y << "," << z << ")" << endl;
	low = Max( IntVector(0,0,0), IntVector(x, y, z) - depth );
	high = Min( arraySize - IntVector(1,1,1), IntVector(x, y, z) + depth );
	int temp = 0;
	for (int local_x = low.x(); local_x <= high.x(); local_x++) {
	  for (int local_y = low.y(); local_y <= high.y(); local_y++) {
	    for (int local_z = low.z(); local_z <= high.z(); local_z++) {
	      //cout << "Filter (" << local_x << "," << local_y << "," << local_z << ")" << endl;
	      temp += flaggedCells[IntVector(local_x, local_y, local_z)]*filter[IntVector(local_x-x+depth.x(),local_y-y+depth.y(),local_z-z+depth.z())];
	      fprintf(stderr,"temp += flaggedCells[%d][%d][%d] * filter[%d][%d][%d] = %d x %d;  temp=%d\n",
		      local_x,local_y,local_z,local_x-x+depth.x(),local_y-y+depth.y(),local_z-z+depth.z(),
		      flaggedCells[IntVector(local_x, local_y, local_z)],filter[IntVector(local_x-x+depth.x(),local_y-y+depth.y(),local_z-z+depth.z())],
		      temp);
	    }
	  }
	}
	dilatedFlaggedCells[IntVector(x,y,z)] = static_cast<int>(temp > 0);
	fprintf(stderr,"temp = %d  dilatedFlaggedCells(%d,%d,%d) = %d\n\n",temp,x,y,z,dilatedFlaggedCells[IntVector(x,y,z)]);
      }
    }
  }
  
  cout << "----------------------------------------------------------------" << endl;
  cout << "FLAGGED CELLS" << endl;

  for (int z = 0; z < arraySize.z(); z++) {
    for (int y = 0; y < arraySize.y(); y++) {
      for (int x = 0; x < arraySize.x(); x++) {
	cout << flaggedCells[IntVector(x,y,z)] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }

  cout << "----------------------------------------------------------------" << endl;
  cout << "DILATED FLAGGED CELLS" << endl;

  for (int z = 0; z < arraySize.z(); z++) {
    for (int y = 0; y < arraySize.y(); y++) {
      for (int x = 0; x < arraySize.x(); x++) {
	cout << dilatedFlaggedCells[IntVector(x,y,z)] << " ";
      }
      cout << endl;
    }
    cout << endl;
  }
}
