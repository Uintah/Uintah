#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>

#include <iostream>

using namespace std;
using namespace Uintah;


RegridderCommon::RegridderCommon(const ProcessorGroup* pg) : Regridder(), UintahParallelComponent(pg)
{

}

RegridderCommon::~RegridderCommon()
{

}

bool RegridderCommon::needRecompile(double time, double delt, const GridP& grid)
{
  bool retval = newGrid;
  newGrid = false;
  return retval;
}

void RegridderCommon::problemSetup(const ProblemSpecP& params, 
                                   const GridP& origGrid,
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
  d_cellCreationDilation = 1;
  d_cellDeletionDilation = 1;
  d_minBoundaryCells = 1;

  regrid_spec->get("cell_creation_dilation", d_cellCreationDilation);
  regrid_spec->get("cell_deletion_dilation", d_cellDeletionDilation);
  regrid_spec->get("min_boundary_cells", d_minBoundaryCells);

  const LevelP level0 = origGrid->getLevel(0);
  cell_num.resize(d_maxLevels);
  patch_num.resize(d_maxLevels);
  patch_size.resize(d_maxLevels);

  
  // get level0 resolution
  IntVector low, high;
  level0->findCellIndexRange(low, high);
  cell_num[0] = high-low;
  const Patch* patch = level0->selectPatchForCellIndex(IntVector(0,0,0));
  patch_size[0] = patch->getHighIndex() - patch->getLowIndex();
  patch_num[0] = calculateNumberOfPatches(cell_num[0], patch_size[0]);

  for (int k = 1; k < d_maxLevels; k++) {
    cell_num[k] = cell_num[k-1] * d_cellRefinementRatio[k-1];
    patch_size[k] = patch_size[k-1] * d_cellRefinementRatio[k-1] /
      d_latticeRefinementRatio[k-1];
    patch_num[k] = calculateNumberOfPatches(cell_num[k], patch_size[k]);
  }  

  // right now, assume that all patches are the same size (+/- 1 for boundary)

  vector< vector< SCIRun::IntVector> > cell_error;
  vector< vector< SCIRun::IntVector> > cell_error_create;
  vector< vector< SCIRun::IntVector> > cell_error_delete;
  vector< vector< SCIRun::IntVector> > cell_error_boundary;

  cell_error.resize(d_maxLevels);
  cell_error_create.resize(d_maxLevels);
  cell_error_delete.resize(d_maxLevels);
  cell_error_boundary.resize(d_maxLevels);

  for (int levelIdx = 0; levelIdx < origGrid->numLevels(); levelIdx++) {
    LevelP level = origGrid->getLevel(levelIdx);
    for ( Level::patchIterator patchIter = level->patchesBegin(); patchIter != level->patchesEnd(); patchIter++ ) {
        
      const Patch* patch = *patchIter;
      IntVector l(patch->getCellLowIndex());
      IntVector h(patch->getCellHighIndex());

      constCCVariable<int> refineFlag;
      //dw->get(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch, Ghost::None, 0);

//       for(CellIterator iter(l, h); !iter.done(); iter++){
// 	IntVector idx(*iter);
// 	if (refineFlag[idx])
// 	  cell_error[levelIdx].push_back(idx);
//       }

    }
  }
}

bool RegridderCommon::flagCellsExist(DataWarehouse* dw, Patch* patch)
{
  // get the variable on the zero'th material - that's the material 
  // refine flags should live
  constCCVariable<int> refineFlag;
  dw->get(refineFlag, d_sharedState->get_refineFlag_label(), 0, patch, Ghost::None, 0);

  IntVector l(patch->getCellLowIndex());
  IntVector h(patch->getCellHighIndex());

  for(CellIterator iter(l, h); !iter.done(); iter++){
    IntVector idx(*iter);
    if (refineFlag[idx])
      return true;
  }
  return false;
}

IntVector RegridderCommon::calculateNumberOfPatches(IntVector& cell_num, IntVector& patch_size)
{
  IntVector patch_num = Ceil(Vector(cell_num.x(), cell_num.y(), cell_num.z()) / patch_size);
  IntVector remainder = Mod(cell_num, patch_size);
  IntVector small = And(Less(remainder, patch_size / IntVector(2,2,2)), Greater(remainder,IntVector(0,0,0)));
  
  for ( int i = 0; i < 3; i++ ) {
    if (small[i])
      patch_num[small[i]]--;
  }

  return patch_num;
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

void RegridderCommon::Dilate( IndexList& flaggedCells, IndexList& dilatedflaggedCells, int filterType )
{
  switch (filterType) {
  case FILTER_STAR: {
  }
  case FILTER_BOX: {

  }
    //  default: throw Exception;
  }
}
