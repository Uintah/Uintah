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

#include <CCA/Components/Regridder/TiledRegridder.h>
#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/PatchBVH/PatchBVH.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Timers/Timers.hpp>

#include <iomanip>
#include <cstdio>

using namespace Uintah;
using namespace std;

static DebugStream rgtimes("RGTimes", "Regridder", "Tiled regridder times debug stream", false);

extern DebugStream regrider_dbg;

int
Product( const IntVector &i )
{
    return i[0]*i[1]*i[2];
}

TiledRegridder::TiledRegridder(const ProcessorGroup* pg) : RegridderCommon(pg)
{
}

TiledRegridder::~TiledRegridder()
{
}
//______________________________________________________________________
//

void TiledRegridder::ComputeTiles( vector<IntVector> &tiles, 
                                   const LevelP level, 
                                   IntVector tile_size, 
                                   IntVector cellRefinementRatio)
{
  DataWarehouse *dw=m_scheduler->getLastDW();

  const PatchSubset *ps=m_loadBalancer->getPerProcessorPatchSet(level)->getSubset(d_myworld->myRank());
  int newLevelIndex=level->getIndex()+1;

  //for each patch I own
  for(int p=0;p<ps->size();p++) {

    const Patch *patch=ps->get(p);
    constCCVariable<int> flags;
    dw->get(flags, d_dilatedCellsRegridLabel, 0, patch, Ghost::None, 0);

    //compute fine level patch extents
    IntVector patchLow  = patch->getCellLowIndex()  * cellRefinementRatio;
    IntVector patchHigh = patch->getCellHighIndex() * cellRefinementRatio;

    //compute possible tile index's
    IntVector tileLow  = computeTileIndex(patchLow, tile_size);
    IntVector tileHigh = computeTileIndex(patchHigh,tile_size);
    tileHigh += IntVector(1,1,1);   // **** We must use inclusive loops when looping over the tiles ***
    
    // Bulletproofing:
#if SCI_ASSERTION_LEVEL > 0    
    CCVariable<int> beenSearched;
    dw->allocateTemporary( beenSearched, patch);
    beenSearched.initialize(-9);
#endif   
       

    for (CellIterator ti(tileLow,tileHigh); !ti.done(); ti++){
    
      //compute tile extents
      IntVector cellLow =computeCellLowIndex( *ti,d_numCells[newLevelIndex],tile_size);
      IntVector cellHigh=computeCellHighIndex(*ti,d_numCells[newLevelIndex],tile_size);
      
      //intersect tile and patch
      IntVector searchLow =Max(cellLow, patchLow)/cellRefinementRatio;
      IntVector searchHigh=Min(cellHigh,patchHigh)/cellRefinementRatio;

#if 0
      if(patch->getID() == 222082){
          cout << "   Patch_ID  " << patch->getID() << endl;
          cout << "   coarsePatch : "<< patch->getCellLowIndex() << " patchHigh: " << patch->getCellHighIndex() << endl;
          cout << "   finePatch   : " << patchLow << " finePatchHigh: " << patchHigh << endl;
          cout << "   tileIndex   : " << *ti << " tileLow:  " << tileLow << " tileHigh: " << tileHigh << endl;
          cout << "   cellLow     : " << cellLow << " cellHigh: " << cellHigh << endl;
      } 
#endif

      
      //search the tile for a refinement flag
      for(CellIterator ci(searchLow,searchHigh); !ci.done(); ci++){

#if SCI_ASSERTION_LEVEL == 0               
        if(flags[*ci]){
          tiles.push_back(*ti);
          break;
        }
#else                               // bulletproofing:  set the beenSearched flag
        if(flags[*ci]){
          tiles.push_back(*ti);
        }
        beenSearched[*ci] = 1;
#endif
      }
    }  // tile loop
    
#if SCI_ASSERTION_LEVEL > 0 
    //__________________________________
    //  BULLET PROOFING If all the cells on the coarse level
    // haven't been searched then throw an exception
    int count = 0;
    for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      if(beenSearched[c] == -9){
        count +=1;
      }
    } 
    
    if(count != 0){
      std::ostringstream msg;
      msg << " ERROR:  TiledRegridder:  Did not search this patch "
          << *patch << " entirely for refinement flags.  Number of cells not searched: " << count << endl;

      throw InternalError(msg.str(),__FILE__,__LINE__);
    }
#endif    
    
  }  // patch loop
}
//______________________________________________________________________
//
Grid* TiledRegridder::regrid(Grid* oldGrid, const int timeStep)
{
  double rtimes[20]={0};

  if(rgtimes.active()) {
    for(int i=0;i<20;i++)
      rtimes[i]=0;
  }

  vector< vector<IntVector> > tiles(min(oldGrid->numLevels()+1,d_maxLevels));

  Timers::Simple timer;
  timer.start();

  //for each level fine to coarse 
  for(int l=min(oldGrid->numLevels()-1,d_maxLevels-2); l >= 0; l--) {
    //Uintah::MPI::Barrier(d_myworld->getComm());

    // THE TIMES FOR THE LEVEL PROVIDE NO INFORMATION OTHER THAN THE
    // TIME TO CHECK THE FOR LOOP.
    rtimes[15+l] += timer().seconds();
    timer.reset( true );
    const LevelP level=oldGrid->getLevel(l);

    vector<IntVector> mytiles;
    vector<IntVector> myoldtiles;

    rtimes[0] += timer().seconds();
    timer.reset( true );
    
    //compute volume using minimum tile size
    ComputeTiles(mytiles, level, d_minTileSize[l+1], d_cellRefinementRatio[l]);
    rtimes[1] += timer().seconds();
    timer.reset( true );

    GatherTiles(mytiles,tiles[l+1]);

    if(l>0) {
      //add flags to the coarser level to ensure that boundary layers exist and that fine patches have a coarse patches above them.
      CoarsenFlags(oldGrid,l,tiles[l+1]);
    }
    rtimes[6] += timer().seconds();
    timer.reset( true );
  }

  //level 0 does not change so just copy the patches over.
  for (Level::const_patch_iterator p = oldGrid->getLevel(0)->patchesBegin(); p != oldGrid->getLevel(0)->patchesEnd(); p++) {
    tiles[0].push_back(computeTileIndex((*p)->getCellLowIndex(),d_tileSize[0]));
  }

  //Create the grid
  Grid *newGrid = CreateGrid(oldGrid,tiles);

  rtimes[7] += timer().seconds();
  timer.reset( true );

  if(*newGrid==*oldGrid) {
    delete newGrid;
    return oldGrid;
  }
  rtimes[8] += timer().seconds();
  timer.reset( true );

  //finalize the grid
  IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
  for(int l=0;l<newGrid->numLevels();l++) {
    LevelP level= newGrid->getLevel(l);
    level->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    //level->assignBCS(grid_ps_,0);
  }

  rtimes[9] += timer().seconds();
  timer.reset( true );

  d_newGrid = true;
  d_lastRegridTimestep = timeStep;

  OutputGridStats(newGrid);

  //initialize the weights on new patches
  m_loadBalancer->initializeWeights(oldGrid,newGrid);

  rtimes[10] += timer().seconds();
  timer.reset( true );
#if SCI_ASSERTION_LEVEL > 0
  if(!verifyGrid(newGrid))
  {
    throw InternalError("Grid is not consistent across processes",__FILE__,__LINE__);
  }
#endif 
  rtimes[11] += timer().seconds();
  timer.reset( true );

  // ignore...
  if (rgtimes.active()) {
    double avg[20] = { 0 };
    Uintah::MPI::Reduce(rtimes, avg, 20, MPI_DOUBLE, MPI_SUM, 0, d_myworld->getComm());
    if (d_myworld->myRank() == 0) {
      cout << "Regrid Avg Times: ";
      for (int i = 0; i < 20; i++) {
        avg[i] /= d_myworld->nRanks();
        cout << i << ":" << avg[i] << " ";
      }
      cout << endl;
    }
    double max[20] = { 0 };
    Uintah::MPI::Reduce(rtimes, max, 20, MPI_DOUBLE, MPI_MAX, 0, d_myworld->getComm());
    if (d_myworld->myRank() == 0) {
      cout << "Regrid Max Times: ";
      for (int i = 0; i < 20; i++) {
        cout << i << ":" << max[i] << " ";
      }
      cout << endl;
    }
  }

  return newGrid;
}
//______________________________________________________________________
//
Grid* TiledRegridder::CreateGrid(Grid* oldGrid, vector<vector<IntVector> > &tiles )
{
  Grid* newGrid = scinew Grid();
  Vector spacing = oldGrid->getLevel(0)->dCell();
  Point anchor =   oldGrid->getLevel(0)->getAnchor();
  IntVector extraCells = oldGrid->getLevel(0)->getExtraCells();

  //For each level Coarse -> Fine
  for(int l=0; l < oldGrid->numLevels()+1 && l < d_maxLevels;l++) {
    // if level is not needed, don't create any more levels
    if(tiles[l].size()==0) {
       break;
    }

    LevelP level = newGrid->addLevel(anchor, spacing);
    level->setExtraCells(extraCells);

    //cout << "New level " << l << " num patches " << patch_sets[l-1].size() << endl;
    //for each patch
    for(unsigned int p=0;p<tiles[l].size();p++) {
      IntVector low = computeCellLowIndex(  tiles[l][p],d_numCells[l],d_tileSize[l]);
      IntVector high = computeCellHighIndex(tiles[l][p],d_numCells[l],d_tileSize[l]);
      //cout << "level: " << l << " Creating patch from tile " << tiles[l][p] << " at " << low << ", " << high << endl;
      //cout << "     numCells: " << d_numCells[l] << " tileSize: " << d_tileSize[l] << endl;

      //create patch
      level->addPatch(low, high, low, high,newGrid);
    }
    // parameters based on next-fine level.
    spacing = spacing / d_cellRefinementRatio[l];
  }
  
  return newGrid;
}

//______________________________________________________________________
//
void TiledRegridder::OutputGridStats(Grid* newGrid)
{
  if (d_myworld->myRank() == 0) 
  {
    cout << " Grid Statistics:\n";
    for (int l = 0; l < newGrid->numLevels(); l++) 
    {
      LevelP level=newGrid->getLevel(l);
      unsigned int num_patches=level->numPatches();
      if(num_patches==0)
        break;

      double total_cells=0;
      double sum_of_cells_squared=0;
      //calculate total cells and cells squared
      for(unsigned int p=0;p<num_patches;p++)
      {
        const Patch* patch=level->getPatch(p);
        double cells=double(patch->getNumCells());
        total_cells+=cells;
        sum_of_cells_squared+=cells*cells;
      }
      
      //calculate conversion factor into simulation coordinates
      double factor=1;
      for(int d=0;d<3;d++)
      {
        factor*=newGrid->getLevel(l)->dCell()[d];
      }
      
      //calculate mean
      double mean = total_cells /(double) num_patches;
      double stdv = sqrt((sum_of_cells_squared-total_cells*total_cells/(double)num_patches)/(double)num_patches);
      IntVector refineRatio = level->getRefinementRatio();
      
      cout << left << "  L" << setw(6)  << l+1 << " RefineRatio: "<<  refineRatio << setw(6)
           << " Patches: " << setw(6) << num_patches << " Total Cells: " << setw(6) << total_cells << " Mean Cells: " << setw(6) << mean << " stdv: " << setw(6) << stdv << " relative stdv: " << setw(8) << stdv/mean << " Volume: " << setw(8) << total_cells*factor << endl;
    }
  }
}
//______________________________________________________________________
//
void TiledRegridder::problemSetup(const ProblemSpecP& params, 
                                  const GridP& oldGrid,
                                  const MaterialManagerP& materialManager)
{
  RegridderCommon::problemSetup(params, oldGrid, materialManager);

  ProblemSpecP amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");
  
  if (!regrid_spec) {
    return; // already warned about it in RC::problemSetup
  }
  // get min patch size
  regrid_spec->require("min_patch_size", d_minTileSize);
  int size=d_minTileSize.size();

  //it is not required to specifiy the minimum patch size on each level
  //if every level is not specified reuse the lowest level minimum patch size
  IntVector lastSize = d_minTileSize[size - 1];
  if (size < d_maxLevels) {
    d_minTileSize.reserve(d_maxLevels);
    for (int i = size; i < d_maxLevels-1; i++){
      d_minTileSize.push_back(lastSize);
    }
  }
  
  d_inputMinTileSize = d_minTileSize;
  
  LevelP level=oldGrid->getLevel(0);

  d_numCells.reserve(d_maxLevels);
  IntVector lowIndex, highIndex;
  level->findInteriorCellIndexRange(lowIndex,highIndex);
  d_numCells[0]=highIndex-lowIndex;
  
  for(int l=1;l<d_maxLevels;l++){
    d_numCells[l]=d_numCells[l-1]*d_cellRefinementRatio[l-1];
  }
  
  //calculate the patch size on level 0
  IntVector patch_size(0,0,0);

  for(Level::patch_iterator patch=level->patchesBegin();patch<level->patchesEnd();patch++)
  {
    IntVector size=(*patch)->getCellHighIndex()-(*patch)->getCellLowIndex();
    if( patch_size == IntVector(0,0,0) ) {
      patch_size = size;
    }
  }
  d_minTileSize.insert(d_minTileSize.begin(),patch_size);

  d_tileSize=d_minTileSize;
  
  //set target patches
  if( d_myworld->nRanks() == 1 ) {
    //if there is only 1 processor attempt for minimum number of patches
    target_patches_=1;
  }
  else {
    int patches_per_proc=4;
    regrid_spec->get("patches_per_level_per_proc",patches_per_proc);
    if ( patches_per_proc < 1 ) {
      if ( d_myworld->myRank() == 0 ) {
        cout << "  Bounding patches_per_level_per_proc to [1,infinity]\n";
      }
      patches_per_proc=1;
    }
    target_patches_=patches_per_proc*d_myworld->nRanks();
  }
  
  for (int k = 0; k < d_maxLevels; k++) {
    if (k < (d_maxLevels)) {
      problemSetup_BulletProofing(k);
    }
  }
}

//_________________________________________________________________

void
TiledRegridder::problemSetup_BulletProofing( const int L )
{
  RegridderCommon::problemSetup_BulletProofing(L);

  // For 2D problems the lattice refinement ratio 
  // and the cell refinement ratio must be 1 in that plane
  for(int dir = 0; dir <3; dir++){
    if(d_cellRefinementRatio[L][dir]%2!=0 && d_cellRefinementRatio[L][dir]!=1)
    {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The specified cell refinement ratio is not divisible by 2\n";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }
    
    if(L!=0 && d_cellNum[L][dir] == 1 && d_minTileSize[L][dir] != 1) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is <3D. \n"
        << " The min Patch Size must be 1 in the other dimensions. \n"
        << "Grid Size: " << d_cellNum[L] 
        << " min patch size: " << d_minTileSize[L] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }

    if(L!=0 && d_cellNum[L][dir] != 1 && d_minTileSize[L][dir] < 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be greater than 4 cells in each dimension \n"
        << "except for 1-cell-wide dimensions.\n"
        << "  Patch size on level " << L << ": " << d_minTileSize[L] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
    
    if( log(d_cellNum[L][dir]/d_minTileSize[L][dir] )>10) {
      ostringstream msg;
      msg << "Problem Setup: CompressedIntVector requires more than 10 bits, the size of the CompressedIntVector needs to be increased";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }

    if( L < d_maxLevels-1 && ( d_inputMinTileSize[L][dir]%d_cellRefinementRatio[L][dir] != 0 ) ) {
      ostringstream msg;
      msg << "Problem Setup: Regridder L-"<< L <<": The min_patch_size (" << d_inputMinTileSize[L] << ") is not divisible by the cell_refinement_ratio ("
          <<  d_cellRefinementRatio[L] << ")";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }
  }  // loop over directions
  
  
  if ( L!=0 && Mod( d_cellNum[L], d_minTileSize[L] ) != IntVector(0,0,0) ) {
    ostringstream msg;
    msg << "Problem Setup: Regridder: The overall number of cells on level " << L << "(" << d_cellNum[L] << ") is not divisible by the minimum patch size (" <<  d_minTileSize[L] << ")\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }
}

//______________________________________________________________________
//Create flags on level l-1 where ever tiles exist on level l+1 with boundary layers
void TiledRegridder::CoarsenFlags(GridP oldGrid, int l, vector<IntVector> tiles)
{
  ASSERT(l-1>=0);

  DataWarehouse *dw=m_scheduler->getLastDW();
  LevelP level=oldGrid->getLevel(l-1); //get level above this level to write flags to
  const PatchSubset *cp=m_loadBalancer->getPerProcessorPatchSet(level)->getSubset(d_myworld->myRank());

  if (cp->size() == 0)
    return;


  //cout << d_myworld->myRank() << " SAFETY LAYER Level:" << l << " coarse patches:" << cp->size() << " tiles:" << tiles.size() << endl;
  //create a range tree out of my patches
  PatchBVH pbvh(cp->getVector());
  
  //for each tile
  for(unsigned t=0;t<tiles.size();t++)
  {
    //cout << d_myworld->myRank() << "    fine tile: low:" << tiles[t] << " high:" << tiles[t]+d_tileSize[l+1] << endl;

    //add a boundary and convert coordinates to a coarse level by dividing by the refinement ratios.  
    IntVector low = (computeCellLowIndex(tiles[t],d_numCells[l+1],d_tileSize[l+1]) - d_minBoundaryCells)/d_cellRefinementRatio[l]/d_cellRefinementRatio[l-1]; 
    
    IntVector high = Ceil( (computeCellHighIndex(tiles[t],d_numCells[l+1],d_tileSize[l+1]) + d_minBoundaryCells).asVector()
                            / d_cellRefinementRatio[l].asVector() / d_cellRefinementRatio[l-1].asVector()
                         ); 
    //cout << "level " << l << " coarsening flags for tile: " << tiles[t] << " low:" << computeCellLowIndex(tiles[t],d_numCells[l+1],d_tileSize[l+1]) << " high: " << computeCellHighIndex(tiles[t],d_numCells[l+1],d_tileSize[l+1]) << endl;
    //cout << "     coarseLow: " << low << " coarseHigh: " << high << endl;
    //clamp low and high points to domain boundaries 
    for(int d=0;d<3;d++)
    {
      if(low[d]<0)
      {
         low[d]=0;
      }
      if(high[d]>d_cellNum[l-1][d])
      {
          high[d]=d_cellNum[l-1][d];
      }
    }
    //cout << d_myworld->myRank() << "    coarse tile low:" << low << " high:" << high << endl;

    Level::selectType intersecting_patches;
    //intersect range tree
    pbvh.query(low, high, intersecting_patches);
     
    //for each intersecting patch
    for (unsigned int i = 0; i < intersecting_patches.size(); i++)
    {
      const Patch* patch = intersecting_patches[i];
      //cout << d_myworld->myRank() << "         coarse patch:" << *patch << endl;

      //get the flags variable
      CCVariable<int> flags;
      dw->getModifiable(flags, d_dilatedCellsRegridLabel, 0, patch);

      //intersect tile and coarse patch
      IntVector int_low  = Max(patch->getExtraCellLowIndex(), low);
      IntVector int_high = Min(patch->getExtraCellHighIndex(), high);
      
      //cout << d_myworld->myRank() << "             int_low:" << int_low << " int_high:" << int_high << endl;
      //for each intesecting cells
      for (CellIterator iter(int_low, int_high); !iter.done(); iter++)
      {
        //if(flags[*iter]==false)
        //  cout << d_myworld->myRank() << "                changing flag at:" << *iter << " to true\n";
        //set the refinement flag to true
        flags[*iter]=true;
      }
    }
  }
}
//______________________________________________________________________
//
bool TiledRegridder::verifyGrid(Grid *grid)
{
  //if we are running in serial there is no reason to verify that each processor has the same grid.
  if(d_myworld->nRanks()==1)
    return true;

  vector<int> checksums;
  vector<int> their_checksums;
  vector<string> labels;

  int num_levels=grid->numLevels();
  regrider_dbg << d_myworld->myRank() << " Grid number of levels:" << num_levels << endl;
  their_checksums.resize(d_myworld->nRanks());
  Uintah::MPI::Gather(&num_levels,1,MPI_INT,&their_checksums[0],1,MPI_INT,0,d_myworld->getComm());

  if(d_myworld->myRank()==0)
  {
    for(int i=0;i<d_myworld->nRanks();i++)
    {
      if(num_levels!=their_checksums[i])
      {
        cout << d_myworld->myRank() << " Error number of levels does not match on rank " << i << " my levels:" << num_levels << " their levels:" << their_checksums[i] << endl;
        return false;
      }
    }
  }
  
  for(int i=0;i<num_levels;i++)
  {
    LevelP level=grid->getLevel(i);
    checksums.push_back(level->numPatches());
    char label[100];
    sprintf(label,"Patchset on level %d",i);
    labels.push_back(label);

    IntVector Sum;
    IntVector Diff;
    int sum=0;
    int diff=0;
    for(int p=0;p<level->numPatches();p++)
    {
      const Patch* patch = level->getPatch(p); 
      regrider_dbg << d_myworld->myRank() << "    Level: " << i << " Patch " << p << ": " << *patch << endl;
      Sum =Abs(patch->getCellHighIndex()) + Abs(patch->getCellLowIndex());
      Diff=Abs(patch->getCellHighIndex()) - Abs(patch->getCellLowIndex());
      
      sum+=Sum[0]*Sum[1]*Sum[2]*(p+1);
      diff+=Diff[0]*Diff[1]*Diff[2]*(p+1000000);
      //cout << d_myworld->myRank() << " patch:" << *patch << " sum:" << Sum[0]*Sum[1]*Sum[2]*(p+1) << " diff:" << Diff[0]*Diff[1]*Diff[2]*(p+1) << endl;
    }
    checksums[i]+=(sum+diff);
  }

  their_checksums.resize(checksums.size()*d_myworld->nRanks());
  Uintah::MPI::Gather(&checksums[0],checksums.size(),MPI_INT,&their_checksums[0],checksums.size(),MPI_INT,0,d_myworld->getComm());
 
  if(d_myworld->myRank()==0)
  {
    for(int p=0;p<d_myworld->nRanks();p++)
    {
      for(unsigned int i=0;i<checksums.size();i++)
      {
        if(checksums[i]!=their_checksums[p*checksums.size()+i])
        {
          cout << d_myworld->myRank() << " Error grid inconsistency: " << labels[i] << " does not match on rank:" << p << endl;
          return false;
        }
      }
    }
  }
  //if(d_myworld->myRank()==0)
  //  cout << " GRIDS ARE CONSISTENT\n";
  return true;
}

//______________________________________________________________________
//
//maps a cell index to a tile index
IntVector TiledRegridder::computeTileIndex(const IntVector& cellIndex, const IntVector& tileSize)
{
  return cellIndex/tileSize;

}
//______________________________________________________________________
//
//maps a tile index to the cell low index for that tile
IntVector TiledRegridder::computeCellLowIndex(const IntVector& tileIndex, const IntVector& numCells, const IntVector& tileSize)
{
  IntVector numPatches=numCells/tileSize;
  return numCells*tileIndex/numPatches;
}
//______________________________________________________________________
//
//maps a tile index to the cell high index for that tile
IntVector TiledRegridder::computeCellHighIndex(const IntVector& tileIndex, const IntVector& numCells, const IntVector& tileSize)
{
  return computeCellLowIndex(tileIndex+IntVector(1,1,1),numCells,tileSize);
}
//______________________________________________________________________
//
struct CompressedIntVector
{
  unsigned int x : 10;
  unsigned int y : 10;
  unsigned int z : 10;
  int operator[](int index)
  {
    switch (index)
    {
      case 0:
        return x;
      case 1:
        return y;
      case 2:
        return z;
      default:
        throw InternalError("CompressedIntVector invalid index",__FILE__,__LINE__);
    }
  }
};
//______________________________________________________________________
//
void TiledRegridder::GatherTiles(vector<IntVector>& mytiles, vector<IntVector> &gatheredTiles )
{
  set<IntVector> settiles;
  if(d_myworld->nRanks()>1)
  {
    unsigned int mycount=mytiles.size();
    vector<unsigned int> counts(d_myworld->nRanks());

    vector<CompressedIntVector> tiles(mytiles.size()), gtiles;

    //copy tiles into compressed data structure
    for(size_t i=0; i<tiles.size();i++)
    {
      tiles[i].x=mytiles[i].x();
      tiles[i].y=mytiles[i].y();
      tiles[i].z=mytiles[i].z();
    }
   
    //gather the number of tiles on each processor
    Uintah::MPI::Allgather(&mycount,1,MPI_UNSIGNED,&counts[0],1,MPI_UNSIGNED,d_myworld->getComm());

    //compute the displacements and recieve counts for a gatherv
    vector<int> displs(d_myworld->nRanks());
    vector<int> recvcounts(d_myworld->nRanks());

    int pos=0;
    for(int p=0;p<d_myworld->nRanks();p++)
    {
      displs[p]=pos;
      recvcounts[p]=counts[p]*sizeof(CompressedIntVector);
      //cout << d_myworld->myRank() << " displs: " << displs[p] << " recvs: " << recvcounts[p] << endl;
      pos+=recvcounts[p];
    }

    gtiles.resize(pos/sizeof(CompressedIntVector));

    //gatherv tiles
    Uintah::MPI::Allgatherv(&tiles[0],recvcounts[d_myworld->myRank()], MPI_BYTE, &gtiles[0], &recvcounts[0],
                   &displs[0], MPI_BYTE, d_myworld->getComm());
    
    //tiles might not be unique so add them to a set to make them unique
    //copy compressed tiles into the tile set
    for(size_t i=0;i<gtiles.size();i++)
      settiles.insert(IntVector(gtiles[i].x,gtiles[i].y,gtiles[i].z));

    gtiles.clear();
  }
  else
  {
    //tiles might not be unique so add them to a set to make them unique
    for(size_t i=0;i<mytiles.size();i++)
      settiles.insert(mytiles[i]);
  }
    
  //reassign set to a vector
  gatheredTiles.assign(settiles.begin(),settiles.end());

}
