#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>

using namespace Uintah;

RegridderCommon::RegridderCommon(ProcessorGroup* pg) : Regridder(), UintahParallelComponent(pg)
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
                                   const GridP& origGrid)
{
  // get max num levels

  // get cell refinement ratio, expand it to max levels

  // get lattice refinement ratio, expand it to max levels

  // get other init parameters


  Level* l0 = oldgrid->getLevel(0);
  cell_num.resize(d_maxLevels);
  patch_num.resize(d_maxLevels);
  patch_size.resize(d_maxLevels);

  
  // get l0 resolution
  IntVector low, high;
  l0->findCellIndexRange(low, high);
  cell_num[0] = (high-low) / l0->dCell();
  patch_size[0] = l0->getPatch(0)->getHighIndex() - l0->getPatch()->getLowIndex();
  patch_num[0] = 

  for (int k = 1; k < d_maxLevels(); k++) {
    cell_num[k] = cell_num[k-1] * d_cellRefinementRatio[k-1];
    patch_size[k] = patch_size[k-1] * d_cellRefinementRatio[k-1] /
      d_latticeRefinementRatio[k-1];
    patch_num[k] = 
  }  

  // right now, assume that all patches are the same size (+/- 1 for boundary)
  IntVector patch_size = l0->getPatch(0)->getHighIndex - l0->getPatch(0)->getLowIndex();

  
  

}

bool flagCellsExist(DataWarehouse* dw, Patch* patch)
{
  // get the variable on the zero'th material - that's the material 
  // refine flags should live
  CCVariable<int> refineFlag = dw->get(refineFlag, 
                                       sharedState_->get_refineFlag_label(),
                                       0, patch, Ghost::None, 0);

  IntVector l(patch->getCellLowIndex();
  IntVector h(patch->getcellHighIndex();

  for(CellIterator iter(l, h); !iter.done(); iter++){
    IntVector idx(*iter);
    if (refineFlag[idx])
      return true;
  }
  return false;
}

int calculateNumberOfPatches(IntVector cell_num, IntVector patch_size)
{
  IntVector patch_num = ceil(cell_num / patch_size);
  IntVector remainder = cell_num % patch_size;
  IntVector small = find((remainder < patch_size / 2) & (remainder > 0));
  
  for ( int i = 0; i < small.size(); i++ )
}

