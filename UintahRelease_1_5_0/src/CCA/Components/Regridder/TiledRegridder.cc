/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
//Allgatherv currently performs poorly on Kraken.  
//This hack changes the Allgatherv to an allgather 
//by padding the digits
//#define AG_HACK  



#include <TauProfilerForSCIRun.h>
#include <CCA/Components/Regridder/TiledRegridder.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <Core/Grid/PatchBVH/PatchBVH.h>
#include <Core/Thread/Time.h>
using namespace Uintah;

#include <iomanip>
#include <cstdio>
using namespace std;

static DebugStream grid_dbg("GridDBG",false);
static DebugStream rgtimes("RGTimes",false);

int Product(const IntVector &i)
{
    return i[0]*i[1]*i[2];
}

TiledRegridder::TiledRegridder(const ProcessorGroup* pg) : RegridderCommon(pg)
{

}

TiledRegridder::~TiledRegridder()
{
}


void TiledRegridder::ComputeTiles(vector<IntVector> &tiles, const LevelP level, IntVector tile_size, IntVector cellRefinementRatio)
{
  DataWarehouse *dw=sched_->getLastDW();

  const PatchSubset *ps=lb_->getPerProcessorPatchSet(level)->getSubset(d_myworld->myrank());
  int newLevelIndex=level->getIndex()+1;
  //for each patch I own
  for(int p=0;p<ps->size();p++)
  {
    const Patch *patch=ps->get(p);
    constCCVariable<int> flags;
    dw->get(flags, d_dilatedCellsRegridLabel, 0, patch, Ghost::None, 0);
    
    //compute patch extents
    IntVector patchLow=patch->getCellLowIndex()*cellRefinementRatio;
    IntVector patchHigh=patch->getCellHighIndex()*cellRefinementRatio;

    //compute possible tile index's
    IntVector tileLow=computeTileIndex(patchLow,d_numCells[newLevelIndex],tile_size);
    IntVector tileHigh=computeTileIndex(patchHigh,d_numCells[newLevelIndex],tile_size);

    //cout << "patchLow: " << patchLow << " patchHigh: " << patchHigh << endl;
    //cout << "tileLow: " << tileLow << " tileHigh: " << tileHigh << endl;
    for (CellIterator ti(tileLow,tileHigh); !ti.done(); ti++)
    {
      //compute tile extents
      IntVector cellLow=computeCellLowIndex(*ti,d_numCells[newLevelIndex],tile_size);
      IntVector cellHigh=computeCellHighIndex(*ti,d_numCells[newLevelIndex],tile_size);
      //cout << "cellLow: " << cellLow << " cellHigh: " << cellHigh << endl;
      //intersect tile and patch
      IntVector searchLow=Max(cellLow,patchLow)/cellRefinementRatio;
      IntVector searchHigh=Min(cellHigh,patchHigh)/cellRefinementRatio;
      //cout << "searchLow: " << searchLow << " searchHigh: " << searchHigh << endl;
      //search intersection for a refinement flag
      for(CellIterator ci(searchLow,searchHigh); !ci.done(); ci++)
      {
        if(flags[*ci])
        {
          //cout << "Flag found on level " << newLevelIndex-1 << " at: " << *ci << " adding tile: " << *ti << endl;
          //add tile to the list
          tiles.push_back(*ti);
          break;
        }
      }
    }
  }
}
double rtimes[20]={0};

Grid* TiledRegridder::regrid(Grid* oldGrid)
{
  if(rgtimes.active())
  {
    for(int i=0;i<20;i++)
      rtimes[i]=0;
  }
  double start=Time::currentSeconds();
  MALLOC_TRACE_TAG_SCOPE("TiledRegridder::regrid");
  TAU_PROFILE("TiledRegridder::regrid", " ", TAU_USER);

  vector< vector<IntVector> > tiles(min(oldGrid->numLevels()+1,d_maxLevels));

  //for each level fine to coarse 
  for(int l=min(oldGrid->numLevels()-1,d_maxLevels-2); l >= 0;l--)
  {
    //MPI_Barrier(d_myworld->getComm());
    rtimes[15+l]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();
    const LevelP level=oldGrid->getLevel(l);

    vector<IntVector> mytiles;
    vector<IntVector> myoldtiles;

    rtimes[0]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();
    //compute volume using minimum tile size
    ComputeTiles(mytiles,level,d_minTileSize[l+1],d_cellRefinementRatio[l]);
    rtimes[1]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();

    GatherTiles(mytiles,tiles[l+1]);

    if(l>0) 
    {
      //add flags to the coarser level to ensure that boundary layers exist and that fine patches have a coarse patches above them.
      CoarsenFlags(oldGrid,l,tiles[l+1]);
    }
    rtimes[6]+=Time::currentSeconds()-start;
    start=Time::currentSeconds();
  }

  //level 0 does not change so just copy the patches over.
  for (Level::const_patchIterator p = oldGrid->getLevel(0)->patchesBegin(); p != oldGrid->getLevel(0)->patchesEnd(); p++)
  {
    tiles[0].push_back(computeTileIndex((*p)->getCellLowIndex(),d_numCells[0],d_tileSize[0]));
  }

  //Create the grid
  Grid *newGrid = CreateGrid(oldGrid,tiles);

  rtimes[7]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();

  if(*newGrid==*oldGrid)
  {
    delete newGrid;
    return oldGrid;
  }
  rtimes[8]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();

  //finalize the grid
  TAU_PROFILE_TIMER(finalizetimer, "TiledRegridder::finalize grid", "", TAU_USER);
  TAU_PROFILE_START(finalizetimer);
  IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();

  for(int l=0;l<newGrid->numLevels();l++)
  {
    LevelP level= newGrid->getLevel(l);
    level->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    //level->assignBCS(grid_ps_,0);
  }
  rtimes[9]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
  TAU_PROFILE_STOP(finalizetimer);

  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();

  OutputGridStats(newGrid);


  //initialize the weights on new patches
  lb_->initializeWeights(oldGrid,newGrid);

  rtimes[10]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();
#if SCI_ASSERTION_LEVEL > 0
  if(!verifyGrid(newGrid))
  {
    throw InternalError("Grid is not consistent across processes",__FILE__,__LINE__);
  }
#endif 
  rtimes[11]+=Time::currentSeconds()-start;
  start=Time::currentSeconds();


  if(rgtimes.active())
  {
    double avg[20]={0};
    MPI_Reduce(&rtimes,&avg,20,MPI_DOUBLE,MPI_SUM,0,d_myworld->getComm());
    if(d_myworld->myrank()==0) {
      cout << "Regrid Avg Times: ";
      for(int i=0;i<20;i++)
      {
        avg[i]/=d_myworld->size();
        cout << i << ":" << avg[i] << " ";
      }
      cout << endl;
    }
    double max[20]={0};
    MPI_Reduce(&rtimes,&max,20,MPI_DOUBLE,MPI_MAX,0,d_myworld->getComm());
    if(d_myworld->myrank()==0) {
      cout << "Regrid Max Times: ";
      for(int i=0;i<20;i++)
      {
        cout << i << ":" << max[i] << " ";
      }
      cout << endl;
    }
  }

  return newGrid;
}
Grid* TiledRegridder::CreateGrid(Grid* oldGrid, vector<vector<IntVector> > &tiles )
{
  MALLOC_TRACE_TAG_SCOPE("TiledRegridd::CreateGrid");
  TAU_PROFILE("TiledRegridder::CreateGrid()", " ", TAU_USER);
  Grid* newGrid = scinew Grid();
  
  Vector spacing = oldGrid->getLevel(0)->dCell();
  Point anchor = oldGrid->getLevel(0)->getAnchor();
  IntVector extraCells = oldGrid->getLevel(0)->getExtraCells();

  //For each level Coarse -> Fine
  for(int l=0; l < oldGrid->numLevels()+1 && l < d_maxLevels;l++)
  {
    // if level is not needed, don't create any more levels
    if(tiles[l].size()==0)
       break;

    LevelP level = newGrid->addLevel(anchor, spacing);
    level->setExtraCells(extraCells);

    //cout << "New level " << l << " num patches " << patch_sets[l-1].size() << endl;
    //for each patch
    for(unsigned int p=0;p<tiles[l].size();p++)
    {
      IntVector low = computeCellLowIndex(tiles[l][p],d_numCells[l],d_tileSize[l]);
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


void TiledRegridder::OutputGridStats(Grid* newGrid)
{
  if (d_myworld->myrank() == 0) 
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
      cout << left << "  L" << setw(8) << l+1 << ": Patches: " << setw(8) << num_patches << " Total Cells: " << setw(8) << total_cells << " Mean Cells: " << setw(8) << mean << " stdv: " << setw(8) << stdv << " relative stdv: " << setw(8) << stdv/mean << " Volume: " << setw(8) << total_cells*factor << endl;
    }
  }
}

void TiledRegridder::problemSetup(const ProblemSpecP& params, 
                                const GridP& oldGrid,
                                const SimulationStateP& state)
{
  RegridderCommon::problemSetup(params, oldGrid, state);
  d_sharedState = state;

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
    for (int i = size; i < d_maxLevels-1; i++)
      d_minTileSize.push_back(lastSize);
  }
  
  LevelP level=oldGrid->getLevel(0);

  d_numCells.reserve(d_maxLevels);
  IntVector lowIndex, highIndex;
  level->findInteriorCellIndexRange(lowIndex,highIndex);
  d_numCells[0]=highIndex-lowIndex;
  for(int l=1;l<d_maxLevels;l++)
    d_numCells[l]=d_numCells[l-1]*d_cellRefinementRatio[l-1];

  //calculate the patch size on level 0
  IntVector patch_size(0,0,0);

  for(Level::patchIterator patch=level->patchesBegin();patch<level->patchesEnd();patch++)
  {
    IntVector size=(*patch)->getCellHighIndex()-(*patch)->getCellLowIndex();
    if(patch_size==IntVector(0,0,0))
      patch_size=size;
  }
  d_minTileSize.insert(d_minTileSize.begin(),patch_size);

  d_tileSize=d_minTileSize;
  
  //set target patches
  if(d_myworld->size()==1)
  {
    //if there is only 1 processor attempt for minimum number of patches
    target_patches_=1;
  }
  else
  {
    int patches_per_proc=4;
    regrid_spec->get("patches_per_level_per_proc",patches_per_proc);
    if (patches_per_proc<1)
    {
      if (d_myworld->myrank() == 0)
        cout << "  Bounding patches_per_level_per_proc to [1,infinity]\n";
      patches_per_proc=1;
    }
    target_patches_=patches_per_proc*d_myworld->size();
  }
  for (int k = 0; k < d_maxLevels; k++) {
    if (k < (d_maxLevels)) {
      problemSetup_BulletProofing(k);
    }
  }
  
}

//_________________________________________________________________
void TiledRegridder::problemSetup_BulletProofing(const int k)
{
  RegridderCommon::problemSetup_BulletProofing(k);

  // For 2D problems the lattice refinement ratio 
  // and the cell refinement ratio must be 1 in that plane
  for(int dir = 0; dir <3; dir++){
    if(d_cellRefinementRatio[k][dir]%2!=0 && d_cellRefinementRatio[k][dir]!=1)
    {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The specified cell refinement ratio is not divisible by 2\n";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }
    if(k!=0 && d_cellNum[k][dir] == 1 && d_minTileSize[k][dir] != 1) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is <3D. \n"
        << " The min Patch Size must be 1 in the other dimensions. \n"
        << "Grid Size: " << d_cellNum[k] 
        << " min patch size: " << d_minTileSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }

    if(k!=0 && d_cellNum[k][dir] != 1 && d_minTileSize[k][dir] < 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be greater than 4 cells in each dimension \n"
        << "except for 1-cell-wide dimensions.\n"
        << "  Patch size on level " << k << ": " << d_minTileSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
    if (k!=0 && Mod( d_cellNum[k], d_minTileSize[k] ) != IntVector(0,0,0) ) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: The overall number of cells on level " << k << "(" << d_cellNum[k] << ") is not divisible by the minimum patch size (" <<  d_minTileSize[k] << ")\n";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }
    if(log(d_cellNum[k][dir]/d_minTileSize[k][dir])>10)
    {
      ostringstream msg;
      msg << "Problem Setup: CompressedIntVector requires more than 10 bits, the size of the CompressedIntVector needs to be increased";
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }
  }
}

//Create flags on level l-1 where ever tiles exist on level l+1 with boundary layers
void TiledRegridder::CoarsenFlags(GridP oldGrid, int l, vector<IntVector> tiles)
{
  MALLOC_TRACE_TAG_SCOPE("TiledRegridder::CoarsenFlags");
  TAU_PROFILE("TiledRegridder::CoarsenFlags()", " ", TAU_USER);

  ASSERT(l-1>=0);

  DataWarehouse *dw=sched_->getLastDW();
  LevelP level=oldGrid->getLevel(l-1); //get level above this level to write flags to
  const PatchSubset *cp=lb_->getPerProcessorPatchSet(level)->getSubset(d_myworld->myrank());

  if (cp->size() == 0)
    return;


  //cout << d_myworld->myrank() << " SAFETY LAYER Level:" << l << " coarse patches:" << cp->size() << " tiles:" << tiles.size() << endl;
  //create a range tree out of my patches
  PatchBVH pbvh(cp->getVector());
  
  //for each tile
  for(unsigned t=0;t<tiles.size();t++)
  {
    //cout << d_myworld->myrank() << "    fine tile: low:" << tiles[t] << " high:" << tiles[t]+d_tileSize[l+1] << endl;

    //add a boundary and convert coordinates to a coarse level by dividing by the refinement ratios.  
    IntVector low = (computeCellLowIndex(tiles[t],d_numCells[l+1],d_tileSize[l+1])-d_minBoundaryCells)/d_cellRefinementRatio[l]/d_cellRefinementRatio[l-1]; 
    IntVector high = Ceil( (computeCellHighIndex(tiles[t],d_numCells[l+1],d_tileSize[l+1])+d_minBoundaryCells).asVector()
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
    //cout << d_myworld->myrank() << "    coarse tile low:" << low << " high:" << high << endl;

    Level::selectType intersecting_patches;
    //intersect range tree
    pbvh.query(low, high, intersecting_patches);
     
    //for each intersecting patch
    for (int i = 0; i < intersecting_patches.size(); i++)
    {
      const Patch* patch = intersecting_patches[i];
      //cout << d_myworld->myrank() << "         coarse patch:" << *patch << endl;

      //get the flags variable
      CCVariable<int> flags;
      dw->getModifiable(flags, d_dilatedCellsRegridLabel, 0, patch);

      //intersect tile and coarse patch
      IntVector int_low = Max(patch->getExtraCellLowIndex(), low);
      IntVector int_high = Min(patch->getExtraCellHighIndex(), high);
      
      //cout << d_myworld->myrank() << "             int_low:" << int_low << " int_high:" << int_high << endl;
      //for each intesecting cells
      for (CellIterator iter(int_low, int_high); !iter.done(); iter++)
      {
        //if(flags[*iter]==false)
        //  cout << d_myworld->myrank() << "                changing flag at:" << *iter << " to true\n";
        //set the refinement flag to true
        flags[*iter]=true;
      }
    }
  }
}

bool TiledRegridder::verifyGrid(Grid *grid)
{
  //if we are running in serial there is no reason to verify that each processor has the same grid.
  if(d_myworld->size()==1)
    return true;

  vector<int> checksums;
  vector<int> their_checksums;
  vector<string> labels;

  int num_levels=grid->numLevels();
  grid_dbg << d_myworld->myrank() << " Grid number of levels:" << num_levels << endl;
  their_checksums.resize(d_myworld->size());
  MPI_Gather(&num_levels,1,MPI_INT,&their_checksums[0],1,MPI_INT,0,d_myworld->getComm());

  if(d_myworld->myrank()==0)
  {
    for(int i=0;i<d_myworld->size();i++)
    {
      if(num_levels!=their_checksums[i])
      {
        cout << d_myworld->myrank() << " Error number of levels does not match on rank " << i << " my levels:" << num_levels << " their levels:" << their_checksums[i] << endl;
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
      grid_dbg << d_myworld->myrank() << "    Level: " << i << " Patch " << p << ": " << *patch << endl;
      Sum=Abs(patch->getCellHighIndex())+Abs(patch->getCellLowIndex());
      Diff=Abs(patch->getCellHighIndex())-Abs(patch->getCellLowIndex());
      
      sum+=Sum[0]*Sum[1]*Sum[2]*(p+1);
      diff+=Diff[0]*Diff[1]*Diff[2]*(p+1000000);
      //cout << d_myworld->myrank() << " patch:" << *patch << " sum:" << Sum[0]*Sum[1]*Sum[2]*(p+1) << " diff:" << Diff[0]*Diff[1]*Diff[2]*(p+1) << endl;
    }
    checksums[i]+=(sum+diff);
  }

  their_checksums.resize(checksums.size()*d_myworld->size());
  MPI_Gather(&checksums[0],checksums.size(),MPI_INT,&their_checksums[0],checksums.size(),MPI_INT,0,d_myworld->getComm());
 
  if(d_myworld->myrank()==0)
  {
    for(int p=0;p<d_myworld->size();p++)
    {
      for(unsigned int i=0;i<checksums.size();i++)
      {
        if(checksums[i]!=their_checksums[p*checksums.size()+i])
        {
          cout << d_myworld->myrank() << " Error grid inconsistency: " << labels[i] << " does not match on rank:" << p << endl;
          return false;
        }
      }
    }
  }
  //if(d_myworld->myrank()==0)
  //  cout << " GRIDS ARE CONSISTENT\n";
  return true;
}

//maps a cell index to a tile index
IntVector TiledRegridder::computeTileIndex(const IntVector& cellIndex, const IntVector& numCells, const IntVector& tileSize)
{
  return cellIndex/tileSize;

}
//maps a tile index to the cell low index for that tile
IntVector TiledRegridder::computeCellLowIndex(const IntVector& tileIndex, const IntVector& numCells, const IntVector& tileSize)
{
  IntVector numPatches=numCells/tileSize;
  return numCells*tileIndex/numPatches;
}
//maps a tile index to the cell high index for that tile
IntVector TiledRegridder::computeCellHighIndex(const IntVector& tileIndex, const IntVector& numCells, const IntVector& tileSize)
{
  return computeCellLowIndex(tileIndex+IntVector(1,1,1),numCells,tileSize);
}

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
      case 3:
        throw InternalError("CompressedIntVector invalid index",__FILE__,__LINE__);
    }
  }
};

void TiledRegridder::GatherTiles(vector<IntVector>& mytiles, vector<IntVector> &gatheredTiles )
{
  set<IntVector> settiles;
  if(d_myworld->size()>1)
  {
    unsigned int mycount=mytiles.size();
    vector<unsigned int> counts(d_myworld->size());

    vector<CompressedIntVector> tiles(mytiles.size()), gtiles;

    //copy tiles into compressed data structure
    for(size_t i=0; i<tiles.size();i++)
    {
      tiles[i].x=mytiles[i].x();
      tiles[i].y=mytiles[i].y();
      tiles[i].z=mytiles[i].z();
    }
   
    //gather the number of tiles on each processor
    MPI_Allgather(&mycount,1,MPI_UNSIGNED,&counts[0],1,MPI_UNSIGNED,d_myworld->getComm());

    //compute the displacements and recieve counts for a gatherv
    vector<int> displs(d_myworld->size());
    vector<int> recvcounts(d_myworld->size());

    int pos=0;
    for(int p=0;p<d_myworld->size();p++)
    {
      displs[p]=pos;
      recvcounts[p]=counts[p]*sizeof(CompressedIntVector);
      //cout << d_myworld->myrank() << " displs: " << displs[p] << " recvs: " << recvcounts[p] << endl;
      pos+=recvcounts[p];
    }


#ifdef AG_HACK
    //compute the maximum number on all processors
    int max_s=0;
    for(int p=0;p<d_myworld->size();p++)
      if(recvcounts[p]>max_s)
        max_s=recvcounts[p];

    //cout << d_myworld->myrank() << " mysize: " << tiles.size() << endl;
    //cout << d_myworld->myrank() << " Max: " << max_s << endl;

    //resize the local vector
    tiles.resize(max_s/sizeof(CompressedIntVector));
    //resize the global vector
    gtiles.resize(tiles.size()*d_myworld->size());


    //cout << d_myworld->myrank() << " gathering size on each proc: " << max_s << endl;
    //perform allgather
    MPI_Allgather(&tiles[0],max_s,MPI_BYTE,&gtiles[0],max_s,MPI_BYTE,d_myworld->getComm());

    //tiles might not be unique so add them to a set to make them unique
    //copy compressed tiles into the tile set
    for(int p=0;p<d_myworld->size();p++)
    {
      int start=tiles.size()*p;
      int end=start+recvcounts[p]/sizeof(CompressedIntVector);
      //cout << d_myworld->myrank() << " Adding " << start << " to " << end << endl;
      for(int i=start;i<end;i++)
        settiles.insert(IntVector(gtiles[i].x,gtiles[i].y,gtiles[i].z));
    }

#else
    gtiles.resize(pos/sizeof(CompressedIntVector));

    //gatherv tiles
    MPI_Allgatherv(&tiles[0],recvcounts[d_myworld->myrank()],MPI_BYTE,&gtiles[0],&recvcounts[0],&displs[0],MPI_BYTE,d_myworld->getComm());
    
    //tiles might not be unique so add them to a set to make them unique
    //copy compressed tiles into the tile set
    for(size_t i=0;i<gtiles.size();i++)
      settiles.insert(IntVector(gtiles[i].x,gtiles[i].y,gtiles[i].z));
#endif
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
