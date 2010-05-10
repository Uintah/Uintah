/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <TauProfilerForSCIRun.h>
#include <Packages/Uintah/CCA/Components/Regridder/TiledRegridder.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVH.h>
using namespace Uintah;

#include <iomanip>
#include <cstdio>
using namespace std;

static DebugStream grid_dbg("GridDBG",false);

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
  //for each patch I own
  for(int p=0;p<ps->size();p++)
  {
    const Patch *patch=ps->get(p);
    constCCVariable<int> flags;
    dw->get(flags, d_dilatedCellsRegridLabel, 0, patch, Ghost::None, 0);

    //for each fine tile on the fine level
    //multipying by the refinement ratio places the coordinates in the finer level space, dividing by the tile size makes the iterator hit each tile only once
    ASSERT( !((patch->getCellHighIndex__New()-patch->getCellLowIndex__New())*cellRefinementRatio < tile_size))  ;
    for (CellIterator ti(patch->getCellLowIndex__New()*cellRefinementRatio/tile_size, patch->getCellHighIndex__New()*cellRefinementRatio/tile_size); !ti.done(); ti++)
    {
      //compute the starting cells of the tile
      IntVector tile_start_fine=*ti*tile_size;
      IntVector tile_start_coarse=tile_start_fine/cellRefinementRatio;

      //for each coarse flag in tile
      for(CellIterator ci(IntVector(0,0,0),tile_size/cellRefinementRatio); !ci.done(); ci++)
      {
        //if cell contains flag
        if(flags[tile_start_coarse+*ci])
        {
          //add the tile to the finer level
          tiles.push_back(tile_start_fine);
          break;
        }
      }
    }
  }
}
Grid* TiledRegridder::regrid(Grid* oldGrid)
{
  MALLOC_TRACE_TAG_SCOPE("TiledRegridder::regrid");
  TAU_PROFILE("TiledRegridder::regrid", " ", TAU_USER);
 
  vector< vector<IntVector> > tiles(min(oldGrid->numLevels()+1,d_maxLevels));
  
  //for each level fine to coarse 
  for(int l=min(oldGrid->numLevels()-1,d_maxLevels-2); l >= 0;l--)
  {
    const LevelP level=oldGrid->getLevel(l);
    IntVector original_tile_size=d_tileSize[l+1];
    bool retry=true;

    vector<IntVector> mytiles;
    vector<IntVector> myoldtiles;
    IntVector old_tile_size=d_minTileSize[l+1];

    //compute volume using minimum tile size
    ComputeTiles(mytiles,level,d_minTileSize[l+1],d_cellRefinementRatio[l]);

    unsigned int num_patches=0;
    if(d_myworld->size()>1)
    {
      unsigned int mycount=mytiles.size();
  
      //reduce the number of tiles on each processor
      MPI_Allreduce(&mycount,&num_patches,1,MPI_UNSIGNED,MPI_SUM,d_myworld->getComm());
    }
    else
    {
      num_patches=mytiles.size();
    }

    //compute the volume 
    int min_volume=num_patches*Product(d_minTileSize[l+1]);
    
    do
    {
      retry=false;

      //save tiles in case we want to restore them later
      myoldtiles.swap(mytiles); 

      //erase old tileset
      mytiles.resize(0);

      //compute tiles using the new tile size
      //this could be computed form the minimum tile set instead
      ComputeTiles(mytiles,level,d_tileSize[l+1],d_cellRefinementRatio[l]);

      unsigned int num_patches=0;
      if(d_myworld->size()>1)
      {
        unsigned int mycount=mytiles.size();

        //reduce the number of tiles on each processor
        MPI_Allreduce(&mycount,&num_patches,1,MPI_UNSIGNED,MPI_SUM,d_myworld->getComm());
      }
      else
      {
        num_patches=mytiles.size();
      }

      //compute the volume
      int volume=num_patches*Product(d_tileSize[l+1]);

      //volume huristic to decide if the new tile set is "good"
      if(volume*.95>min_volume) //if increasing the tile size significantly increased the volume
      {
        //restore old tiles 
        mytiles.swap(myoldtiles); 
        d_tileSize[l+1]=old_tile_size;
      }
      else 
      {
        old_tile_size=d_tileSize[l+1];

        if(num_patches<target_patches_) //decrease tile size
        {
          //decrease tile size

          //sort the current tile size largest to smallest
          int dims[3]={0,1,2};

          //simple unrolled bubble sort
          if(dims[1]>dims[0])
            swap(dims[0],dims[1]);
          if(dims[2]>dims[1])
            swap(dims[1],dims[2]);
          if(dims[1]>dims[0])
            swap(dims[0],dims[1]);

          //loop through each dimension and take the first one that can be decreased
          for(int d=0;d<3;d++)
          {
            int new_size=d_tileSize[l+1][dims[d]]/2;
            if(new_size>=d_minTileSize[l+1][dims[d]])
            {
              d_tileSize[l+1][dims[d]]=new_size;

              //if(d_myworld->myrank()==0)
              //  cout << " Decreasing tile size on level " << l+1 << " to " << d_tileSize[l+1] << endl;

              retry=true;
              break;
            }
          }
        }
        else if (num_patches>2*target_patches_)
        {
          //increase tile size
          int min_dim=-1;

          //find the smallest non-1 dimension
          for(int d=0;d<3;d++)
          {
            //if dimension is not equal to 1 and is smaller than the other dimensions
            if(d_minTileSize[l+1][d]>1 && (min_dim==-1 || d_tileSize[l+1][d]<d_tileSize[l+1][min_dim]))
            {
              //don't allow tiles to be bigger than the coarser tile
              if(d_tileSize[l+1][d]*2<=d_tileSize[l][d]*d_cellRefinementRatio[l][d])
                min_dim=d;
            }
          }

          if(min_dim!=-1)
          {
            //increase that dimension by the min_tile_size
            d_tileSize[l+1][min_dim]*=2;
            //if(d_myworld->myrank()==0)
            //  cout << " Increasing tile size on level " << l+1 << " to " << d_tileSize[l+1] << " coarser tile " << d_tileSize[l] << endl;
            retry=true;
          }
        } // end else if(num_patches>2*target_patches_)
      } //end else (volume*.90>min_volume)
    }
    while(retry);

    if(d_myworld->myrank()==0 && !(d_tileSize[l+1]==original_tile_size))
    {
      cout << "Tile size on level:" << l+2 << " changed from " << original_tile_size << " to " << d_tileSize[l+1] << endl;
    }

    if(d_myworld->size()>1)
    {
      unsigned int mycount=mytiles.size();
      vector<unsigned int> counts(d_myworld->size());
      
      //gather the number of tiles on each processor
      MPI_Allgather(&mycount,1,MPI_UNSIGNED,&counts[0],1,MPI_UNSIGNED,d_myworld->getComm());


      //compute the displacements and recieve counts for a gatherv
      vector<int> displs(d_myworld->size());
      vector<int> recvcounts(d_myworld->size());

      int pos=0;
      for(int p=0;p<d_myworld->size();p++)
      {
        displs[p]=pos;
        recvcounts[p]=counts[p]*sizeof(IntVector);
        pos+=recvcounts[p];
      }

      tiles[l+1].resize(pos/sizeof(IntVector));

      //gatherv tiles
      MPI_Allgatherv(&mytiles[0],recvcounts[d_myworld->myrank()],MPI_BYTE,&tiles[l+1][0],&recvcounts[0],&displs[0],MPI_BYTE,d_myworld->getComm());
    }
    else
    {
      tiles[l+1]=mytiles;
    }


    if(l>0) 
    {
      //add flags to the coarser level to ensure that boundary layers exist and that fine patches have a coarse patches above them.
      CoarsenFlags(oldGrid,l,tiles[l+1]);
    }
  }
  
  //level 0 does not change so just copy the patches over.
  for (Level::const_patchIterator p = oldGrid->getLevel(0)->patchesBegin(); p != oldGrid->getLevel(0)->patchesEnd(); p++)
  {
    tiles[0].push_back((*p)->getCellLowIndex__New());
  }
 
  //Create the grid
  Grid *newGrid = CreateGrid(oldGrid,tiles);
  if (newGrid->isSimilar(*oldGrid)) 
  {
    delete newGrid;
    return oldGrid;
  }

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
  TAU_PROFILE_STOP(finalizetimer);
  
  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();
  
  OutputGridStats(newGrid);

  //initialize the weights on new patches
  lb_->initializeWeights(oldGrid,newGrid);
 
#if SCI_ASSERTION_LEVEL > 0
  if(!verifyGrid(newGrid))
  {
    throw InternalError("Grid is not consistent across processes",__FILE__,__LINE__);
  }
#endif 

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
      IntVector low = tiles[l][p];
      IntVector high = low+d_tileSize[l];
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
  
  //calculate the minimum patch size on level 0
  IntVector min_size(INT_MAX,INT_MAX,INT_MAX);

  LevelP level=oldGrid->getLevel(0);

  for(Level::patchIterator patch=level->patchesBegin();patch<level->patchesEnd();patch++)
  {
    IntVector size=(*patch)->getCellHighIndex__New()-(*patch)->getCellLowIndex__New();
    min_size=Min(min_size,size);
  }
  d_minTileSize.insert(d_minTileSize.begin(),min_size);

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

    if(k!=0 && d_cellNum[k][dir] != 1 && d_minTileSize[k][dir] <= 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be greater than 4 cells in each dimension \n"
          << "except for 1-cell-wide dimensions.\n"
          << "  Patch size on level " << k << ": " << d_minTileSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
/*
    if(k!=0 && d_cellNum[k][dir] != 1 && d_minTileSize[k][dir] % d_cellRefinementRatio[k][dir] != 0) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Min Patch Size needs to be divisible by the cell refinement ratio\n"
          << "  Patch size on level " << k << ": " << d_minTileSize[k] 
          << ", refinement ratio on level " << k << ": " << d_cellRefinementRatio[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
 */
  }
  if (k!=0 && Mod( d_cellNum[k], d_minTileSize[k] ) != IntVector(0,0,0) ) {
    ostringstream msg;
    msg << "Problem Setup: Regridder: The overall number of cells on level " << k << "(" << d_cellNum[k] << ") is not divisible by the minimum patch size (" <<  d_minTileSize[k] << ")\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
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
    IntVector low = (tiles[t]-d_minBoundaryCells)/d_cellRefinementRatio[l]/d_cellRefinementRatio[l-1];
    IntVector high = Ceil((tiles[t]+d_tileSize[l+1]+d_minBoundaryCells).asVector()/d_cellRefinementRatio[l].asVector()/d_cellRefinementRatio[l-1].asVector());
    
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
      IntVector int_low = Max(patch->getExtraCellLowIndex__New(), low);
      IntVector int_high = Min(patch->getExtraCellHighIndex__New(), high);
      
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
      Sum=Abs(patch->getCellHighIndex__New())+Abs(patch->getCellLowIndex__New());
      Diff=Abs(patch->getCellHighIndex__New())-Abs(patch->getCellLowIndex__New());
      
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
