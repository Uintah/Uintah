/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */  


#include <CCA/Components/Regridder/SingleLevelRegridder.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/LoadBalancer.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace std;

extern DebugStream regrider_dbg;


SingleLevelRegridder::SingleLevelRegridder(const ProcessorGroup* pg) : TiledRegridder(pg)
{
}

SingleLevelRegridder::~SingleLevelRegridder()
{
}

//______________________________________________________________________
//
void SingleLevelRegridder::problemSetup(const ProblemSpecP& params, 
                                        const GridP& oldGrid,
                                        const MaterialManagerP& materialManager)
{

  RegridderCommon::problemSetup(params, oldGrid, materialManager);

  d_maxLevels = oldGrid->numLevels();
  
  // Compute the refinement ratio (RR).  The regridder's
  // definition of RR differs from the value set in Level.cc
  // Note that Level::d_refinementRatio(0) == 1,1,1 is always the case
  
  d_cellRefinementRatio.resize(d_maxLevels);
  d_cellRefinementRatio[0] = IntVector(1,1,1);  
  
  for(int l=0; l<d_maxLevels-1; l++){

    Vector dx_cur_level   = oldGrid->getLevel(l)->dCell();
    Vector dx_next_level  = oldGrid->getLevel(l+1)->dCell();
    Vector r = (dx_cur_level / dx_next_level ) + Vector(1e-6, 1e-6, 1e-6);
    d_cellRefinementRatio[l] = IntVector((int)r.x(), (int)r.y(), (int)r.z());
  }
  
  
  ProblemSpecP amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");
  
  // compute number of cells and the patch size or tile size on all levels
  d_numCells.reserve( d_maxLevels );
  d_tileSize.reserve( d_maxLevels );
  
  for(int l=0; l<d_maxLevels; l++){
    LevelP level = oldGrid->getLevel(l);
    IntVector lowIndex, highIndex;
    level->findInteriorCellIndexRange( lowIndex,highIndex );

    d_numCells[l] = highIndex-lowIndex;

    const Patch* firstPatch = level->getPatch( 0 );
    
    IntVector low  = firstPatch->getCellLowIndex();
    IntVector high = firstPatch->getCellHighIndex();
    d_tileSize[l] = high-low;
  }
  
  // find the level of interest.
  d_level_index = 0;    // default
  regrid_spec->get("level", d_level_index);
  LevelP level = oldGrid->getLevel( d_level_index );
  
  // Let user change the level of interest patch layout
  // This can be especially useful on restarts where
  // you need to increase the number of coarse level patches
  IntVector new_patch_layout(1,1,1);
  regrid_spec->require("new_patch_layout", new_patch_layout);
  
  IntVector myPatchSize = d_numCells[d_level_index]/new_patch_layout;
  d_tileSize[d_level_index] = myPatchSize;
  
  for (int k = 0; k < d_maxLevels; k++) {
    if (k < (d_maxLevels)) {
      problemSetup_BulletProofing(k);
    }
  }
}


//_________________________________________________________________
void SingleLevelRegridder::problemSetup_BulletProofing(const int L)
{
  RegridderCommon::problemSetup_BulletProofing(L);
}

//______________________________________________________________________
//  Reset the patch layout on the level of interest.  The other level's
//  grid structures will remain constant
// 
Grid* SingleLevelRegridder::regrid(Grid* oldGrid, const int timeStep)
{
  vector< vector<IntVector> > tiles(min(oldGrid->numLevels()+1,d_maxLevels));

  //__________________________________
  //  compute tiles or patches based on user input 
  int minLevel = 0;

  for(int l=d_maxLevels-1; l >= minLevel;l--) {

    // Level of interest  
    if( l == d_level_index) {

      const LevelP level = oldGrid->getLevel(l);
      const PatchSubset *patchSS=m_loadBalancer->getPerProcessorPatchSet(level)->getSubset(d_myworld->myRank());
      vector<IntVector> mytiles;

      // For each patch I own
      for(int p=0; p<patchSS->size(); p++) {

        const Patch* patch=patchSS ->get(p);  

        // Compute patch extents        
        IntVector patchLow  = patch->getCellLowIndex();
        IntVector patchHigh = patch->getCellHighIndex();

        // Compute tile indices
        IntVector tileLow  = TiledRegridder::computeTileIndex( patchLow, d_tileSize[l] );
        IntVector tileHigh = TiledRegridder::computeTileIndex( patchHigh,d_tileSize[l] );

        for (CellIterator ti(tileLow,tileHigh); !ti.done(); ti++){
          mytiles.push_back(*ti);
        }
      }  // patch loop

      TiledRegridder::GatherTiles( mytiles,tiles[l] );
      
    } else {
      // Other levels:
      // The level's patch layout does not change so just copy the patches -> tiles
      for (Level::const_patch_iterator p = oldGrid->getLevel(l)->patchesBegin(); p != oldGrid->getLevel(l)->patchesEnd(); p++){
        IntVector me = TiledRegridder::computeTileIndex((*p)->getCellLowIndex(), d_tileSize[l]);
        tiles[l].push_back( me );
      }
    }
  }
  
  //__________________________________
  //  Create the new grid
  Grid *newGrid = TiledRegridder::CreateGrid(oldGrid,tiles);

  if(*newGrid==*oldGrid){
    delete newGrid;
    return oldGrid;
  }

  // Finalize the grid  
  IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();

  for(int l=0;l<newGrid->numLevels();l++){
    LevelP level= newGrid->getLevel(l);
    level->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
  }

  TiledRegridder::OutputGridStats(newGrid);

  // initialize the weights on new patches
  m_loadBalancer->initializeWeights(oldGrid,newGrid);

#if SCI_ASSERTION_LEVEL > 0
  if(! TiledRegridder::verifyGrid(newGrid) ){
    throw InternalError("Grid is not consistent across processes",__FILE__,__LINE__);
  }
#endif 

  return newGrid;
}
