/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#include <CCA/Components/Regridder/HierarchicalRegridder.h>
#include <CCA/Components/Regridder/PerPatchVars.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Task.h>
#include <Core/Parallel/BufferInfo.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Util/DebugStream.h>
#include <Core/Thread/Mutex.h>
#include <Core/Malloc/Allocator.h>
#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

using namespace std;
using namespace Uintah;
using namespace SCIRun;

extern DebugStream rdbg;
extern DebugStream dilate_dbg;
extern Mutex MPITypeLock;

HierarchicalRegridder::HierarchicalRegridder(const ProcessorGroup* pg) : RegridderCommon(pg)
{
  rdbg << "HierarchicalRegridder::HierarchicalRegridder() BGN" << endl;

  d_activePatchesLabel = VarLabel::create("activePatches",
                             PerPatch<SubPatchFlag>::getTypeDescription());

  rdbg << "HierarchicalRegridder::HierarchicalRegridder() END" << endl;
}

HierarchicalRegridder::~HierarchicalRegridder()
{
  rdbg << "HierarchicalRegridder::~HierarchicalRegridder() BGN" << endl;
  VarLabel::destroy(d_activePatchesLabel);

  for (int k = 0; k < d_maxLevels; k++) {
    delete d_patchActive[k];
    delete d_patchCreated[k];
    delete d_patchDeleted[k];
  }  

  d_patchActive.clear();
  d_patchCreated.clear();
  d_patchDeleted.clear();


  rdbg << "HierarchicalRegridder::~HierarchicalRegridder() END" << endl;
}

void HierarchicalRegridder::problemSetup(const ProblemSpecP& params, 
                                   const GridP& oldGrid,
				   const SimulationStateP& state)

{
  if(d_myworld->myrank()==0)
    cout << " WARNING: The Hierarchical regridder has major performance issues and has been superseeded by the tiled regridder\n";
  rdbg << "HierarchicalRegridder::problemSetup() BGN" << endl;
  RegridderCommon::problemSetup(params, oldGrid, state);
  d_sharedState = state;

  ProblemSpecP amr_spec = params->findBlock("AMR");
  ProblemSpecP regrid_spec = amr_spec->findBlock("Regridder");

  if (!regrid_spec) {
    return; // already warned about it in RC::problemSetup
  }
  // get lattice refinement ratio, expand it to max levels
  regrid_spec->require("lattice_refinement_ratio", d_latticeRefinementRatio);

  int size = (int) d_latticeRefinementRatio.size();
  IntVector lastRatio = d_latticeRefinementRatio[size - 1];
  if (size < d_maxLevels) {
    d_latticeRefinementRatio.resize(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++)
      d_latticeRefinementRatio[i] = lastRatio;
  }

  // get lattice refinement ratio, expand it to max levels
  regrid_spec->get("max_patches_to_combine", d_patchesToCombine);
  size = (int) d_patchesToCombine.size();
  if (size == 0) {
    d_patchesToCombine.push_back(IntVector(1,1,1));
    size = 1;
  }
  lastRatio = d_patchesToCombine[size - 1];
  if (size < d_maxLevels) {
    d_patchesToCombine.resize(d_maxLevels);
    for (int i = size; i < d_maxLevels; i++)
      d_patchesToCombine[i] = lastRatio;
  }

  d_patchNum.resize(d_maxLevels);
  d_patchSize.resize(d_maxLevels);
  d_maxPatchSize.resize(d_maxLevels);
  d_patchActive.resize(d_maxLevels);
  d_patchCreated.resize(d_maxLevels);
  d_patchDeleted.resize(d_maxLevels);

  const LevelP level0 = oldGrid->getLevel(0);
  
  // get level0 resolution
  IntVector low, high;
  level0->findCellIndexRange(low, high);
  const Patch* patch = level0->selectPatchForCellIndex(IntVector(0,0,0));
  d_patchSize[0] = patch->getCellHighIndex() - patch->getCellLowIndex();
  d_maxPatchSize[0] = patch->getCellHighIndex() - patch->getCellLowIndex();
  d_patchNum[0] = calculateNumberOfPatches(d_cellNum[0], d_patchSize[0]);
  d_patchActive[0] = scinew CCVariable<int>;
  d_patchCreated[0] = scinew CCVariable<int>;
  d_patchDeleted[0] = scinew CCVariable<int>;
  d_patchActive[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchCreated[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchDeleted[0]->rewindow(IntVector(0,0,0), d_patchNum[0]);
  d_patchActive[0]->initialize(1);
  d_patchCreated[0]->initialize(0);
  d_patchDeleted[0]->initialize(0);
  
  problemSetup_BulletProofing(0);
  
  for (int k = 1; k < d_maxLevels; k++) {
    d_patchSize[k] = d_patchSize[k-1] * d_cellRefinementRatio[k-1] /
      d_latticeRefinementRatio[k-1];
    d_maxPatchSize[k] = d_patchSize[k] * d_patchesToCombine[k-1]; 
    d_latticeRefinementRatio[k-1];
    d_patchNum[k] = calculateNumberOfPatches(d_cellNum[k], d_patchSize[k]);
    d_patchActive[k] = scinew CCVariable<int>;
    d_patchCreated[k] = scinew CCVariable<int>;
    d_patchDeleted[k] = scinew CCVariable<int>;
    d_patchActive[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchCreated[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchDeleted[k]->rewindow(IntVector(0,0,0), d_patchNum[k]);
    d_patchActive[k]->initialize(0);
    d_patchCreated[k]->initialize(0);
    d_patchDeleted[k]->initialize(0);
    if (k < (d_maxLevels)) {
      problemSetup_BulletProofing(k);
    }
  }
}

//_________________________________________________________________
void HierarchicalRegridder::problemSetup_BulletProofing(const int k)
{
  RegridderCommon::problemSetup_BulletProofing(k);


  for(int dir = 0; dir <3; dir++){
   if(d_latticeRefinementRatio[k][dir] < 1 || d_cellRefinementRatio[k][dir] < 1){
       ostringstream msg;
       msg << "Problem Setup: Regridder:"
       << " The lattice refinement ratio AND the cell refinement ration must be at least 1 in any direction. \n"
       << " lattice refinement ratio: " << d_latticeRefinementRatio[k] 
       << " cell refinement ratio: " << d_cellRefinementRatio[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
    }
  }

  // For 2D problems the lattice refinement ratio 
  // and the cell refinement ratio must be 1 in that plane
  for(int dir = 0; dir <3; dir++){
    if(d_cellNum[k][dir] == 1 && (d_latticeRefinementRatio[k][dir] != 1 || d_cellRefinementRatio[k][dir] != 1) ){
      ostringstream msg;
      msg << "Problem Setup: Regridder: The problem you're running is 2D. \n"
          << " The lattice refinement ratio AND the cell refinement ration must be 1 in that direction. \n"
          << "Grid Size: " << d_cellNum[k] 
          << " lattice refinement ratio: " << d_latticeRefinementRatio[k] 
          << " cell refinement ratio: " << d_cellRefinementRatio[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
      
    }

    if(d_cellNum[k][dir] != 1 && d_patchSize[k][dir] < 4) {
      ostringstream msg;
      msg << "Problem Setup: Regridder: Patches need to be at least 4 cells in each dimension \n"
          << "except for 1-cell-wide dimensions.\n"
          << "  Patch size on level " << k << ": " << d_patchSize[k] << endl;
      throw ProblemSetupException(msg.str(), __FILE__, __LINE__);

    }
  }

  if ( Mod( d_patchSize[k], d_latticeRefinementRatio[k] ) != IntVector(0,0,0) ) {
    ostringstream msg;
    msg << "Problem Setup: Regridder: you've specified a patch size (interiorCellHighIndex() - interiorCellLowIndex()) on a patch that is not divisible by the lattice ratio on level 0 \n"
        << " patch size " <<  d_patchSize[k] << " lattice refinement ratio " << d_latticeRefinementRatio[k] << endl;
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }
}


Grid* HierarchicalRegridder::regrid(Grid* oldGrid)
{
  rdbg << "HierarchicalRegridder::regrid() BGN" << endl;

  if (d_maxLevels <= 1)
    return oldGrid;

  if (!grid_ps_) {
    throw InternalError("HierarchicalRegridder::regrid() Grid section of UPS file not found!", __FILE__, __LINE__);
  }

  // this is for dividing the entire regridding problem into patchwise domains
  DataWarehouse* parent_dw = sched_->getLastDW();
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           parent_dw->setScrubbing(DataWarehouse::ScrubNone);


  SchedulerP tempsched = sched_->createSubScheduler();

  // it's normally unconventional to pass the new_dw in in the old_dw's spot,
  // but we don't even use the old_dw and on the first timestep it could be null 
  // and not initialize the parent dws.
  tempsched->initialize(3, 1);
  tempsched->setParentDWs(parent_dw, parent_dw);

  tempsched->clearMappings();
  tempsched->mapDataWarehouse(Task::ParentOldDW, 0);
  tempsched->mapDataWarehouse(Task::ParentNewDW, 1);
  tempsched->mapDataWarehouse(Task::OldDW, 2);
  tempsched->mapDataWarehouse(Task::NewDW, 3);

  // make sure we have data in our subolddw
  tempsched->advanceDataWarehouse(oldGrid);
  tempsched->advanceDataWarehouse(oldGrid);

  tempsched->get_dw(2)->setScrubbing(DataWarehouse::ScrubNone);
  tempsched->get_dw(3)->setScrubbing(DataWarehouse::ScrubNone);
  

  for ( int levelIndex = 0; levelIndex < oldGrid->numLevels() && levelIndex < d_maxLevels-1; levelIndex++ ) {
  // copy dilation to the "old dw" so mpi copying will work correctly
    const PatchSet* perproc = sched_->getLoadBalancer()->getPerProcessorPatchSet(oldGrid->getLevel(levelIndex));
    perproc->addReference();
    const PatchSubset* psub = perproc->getSubset(d_myworld->myrank());
    MaterialSubset* msub = scinew MaterialSubset;
    msub->add(0);
    DataWarehouse* old_dw = tempsched->get_dw(2);
    old_dw->transferFrom(parent_dw, d_dilatedCellsRegridLabel, psub, msub);
    delete msub;

    if (perproc->removeReference())
      delete perproc;
                       
    // mark subpatches on this level (subpatches represent where patches on the next
    // level will be created).
    Task* mark_task = scinew Task("HierarchicalRegridder::MarkPatches2",
                               this, &HierarchicalRegridder::MarkPatches2);
    mark_task->requires(Task::OldDW, d_dilatedCellsRegridLabel, Ghost::None);
#if 0
    if (d_cellCreationDilation != d_cellDeletionDilation)
      mark_task->requires(Task::OldDW, d_dilatedCellsDeletionLabel, Ghost::None);
#endif

    mark_task->computes(d_activePatchesLabel, d_sharedState->refineFlagMaterials());
    tempsched->addTask(mark_task, oldGrid->getLevel(levelIndex)->eachPatch(), d_sharedState->allMaterials());
  }
  
  tempsched->compile();
  tempsched->execute();
  parent_dw->setScrubbing(ParentNewDW_scrubmode);
  GatherSubPatches(oldGrid, tempsched);
  
  Grid* newGrid = CreateGrid2(oldGrid);  

  //initialize the weights on new patches
  lb_->initializeWeights(oldGrid,newGrid);
  return newGrid;
}


IntVector HierarchicalRegridder::StartCellToLattice ( IntVector startCell, int levelIdx )
{
  return startCell / d_patchSize[levelIdx];
}

void HierarchicalRegridder::MarkPatches2(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  rdbg << "MarkPatches2 BGN\n";
  const Level* oldLevel = getLevel(patches);
  int levelIdx = oldLevel->getIndex();
  IntVector subPatchSize = d_patchSize[levelIdx+1]/d_cellRefinementRatio[levelIdx];
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector startCell = patch->getCellLowIndex();
    IntVector endCell = patch->getCellHighIndex();
    IntVector latticeIdx = StartCellToLattice( startCell, levelIdx );
    IntVector realPatchSize = endCell - startCell; // + IntVector(1,1,1); // RNJ this is incorrect maybe?
    IntVector numSubPatches = realPatchSize / subPatchSize;

    IntVector startidx = latticeIdx * d_latticeRefinementRatio[levelIdx];

    SubPatchFlag* subpatches = scinew SubPatchFlag(startidx, startidx+numSubPatches);
    PerPatch<SubPatchFlagP> activePatches(subpatches);
    
    constCCVariable<int> dilatedCellsRegrid;
    // FIX Deletion - CCVariable<int>* dilatedCellsDeleted;
    new_dw->put(activePatches, d_activePatchesLabel, 0, patch);
    old_dw->get(dilatedCellsRegrid, d_dilatedCellsRegridLabel, 0, patch, Ghost::None, 0);
    
    if (d_cellRegridDilation != d_cellDeletionDilation) {
      //FIX Deletion
      //constCCVariable<int> dcd;
      //old_dw->get(dcd, d_dilatedCellsDeletionLabel, 0, patch, Ghost::None, 0);
      //dilatedCellsDeleted = dynamic_cast<CCVariable<int>*>(const_cast<CCVariableBase*>(dcd.clone()));
    }
    else {
      // Fix Deletion - dilatedCellsDeleted = dilatedCellsRegrid;
    }

    for (CellIterator iter(IntVector(0,0,0), numSubPatches); !iter.done(); iter++) {
      IntVector idx(*iter);
      IntVector startCellSubPatch = startCell + idx * subPatchSize;
      IntVector endCellSubPatch = startCell + (idx + IntVector(1,1,1)) * subPatchSize - IntVector(1,1,1);
      IntVector latticeStartIdx = startidx + idx;
      //IntVector latticeEndIdx = latticeStartIdx + IntVector(1,1,1);
      
//       rdbg << "MarkPatches() startCell         = " << startCell         << endl;
//       rdbg << "MarkPatches() endCell           = " << endCell           << endl;
//       rdbg << "MarkPatches() latticeIdx        = " << latticeIdx        << endl;
//       rdbg << "MarkPatches() realPatchSize     = " << realPatchSize     << endl;
//       rdbg << "MarkPatches() numSubPatches     = " << numSubPatches     << endl;
//       rdbg << "MarkPatches() currentIdx        = " << idx               << endl;
//       rdbg << "MarkPatches() startCellSubPatch = " << startCellSubPatch << endl;
//       rdbg << "MarkPatches() endCellSubPatch   = " << endCellSubPatch   << endl;

      if (flaggedCellsExist(dilatedCellsRegrid, startCellSubPatch, endCellSubPatch)) {
        rdbg << "Marking Active [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
        subpatches->set(latticeStartIdx);
//       } else if (!flaggedCellsExist(*dilatedCellsDeleted, startCellSubPatch, endCellSubPatch)) {
//         // Do we need to check for flagged cells in the children?
//         IntVector childLatticeStartIdx = latticeStartIdx;
//         IntVector childLatticeEndIdx = latticeEndIdx;
//         for (int childLevelIdx = levelIdx+1; childLevelIdx < oldLevel->getGrid()->numLevels(); childLevelIdx++) {
//           for (CellIterator inner_iter(childLatticeStartIdx, childLatticeEndIdx); !inner_iter.done(); inner_iter++) {
//             IntVector inner_idx(*inner_iter);
//             rdbg << "Deleting Active [ " << childLevelIdx << " ]: " << inner_idx << endl;
//             subpatches->clear(inner_idx);
//           }
//           childLatticeStartIdx = childLatticeStartIdx * d_latticeRefinementRatio[childLevelIdx];
//           childLatticeEndIdx = childLatticeEndIdx * d_latticeRefinementRatio[childLevelIdx];
//         }
      } else {
        //        rdbg << "Not Marking or deleting [ " << levelIdx+1 << " ]: " << latticeStartIdx << endl;
      }
    }
  }
  rdbg << "MarkPatches2 END\n";
}

void HierarchicalRegridder::GatherSubPatches(const GridP& oldGrid, SchedulerP& sched)
{
  rdbg << "GatherSubPatches BGN\n";

  LoadBalancer* lb = sched->getLoadBalancer();

  // numLevels -1 because that is the index of the finest level, and
  // maxLevels -2 because we don't care about the subpatches on the 
  //   finest level
  int toplevel = Min(oldGrid->numLevels()-1, d_maxLevels-2);

  // make sure we get the right DW
  DataWarehouse* dw = sched->getLastDW();

  d_patches.clear();
  d_patches.resize(toplevel+2);

  for (int i = toplevel; i >= 0; i--) {
    rdbg << "  gathering on level TOP " << toplevel << " Level " << i << endl;
    const Level* l = oldGrid->getLevel(i).get_rep();
    int levelIdx = l->getIndex();

    int numSubpatches = d_patchNum[i+1].x() * d_patchNum[i+1].y() * d_patchNum[i+1].z();
  
    // place to end up with all subpatches
    vector<SubPatchFlagP> allSubpatches(l->numPatches());
    vector<int> recvbuf(numSubpatches); // buffer to recv data (and hold pointers from allSubpatches)
    if (d_myworld->size() > 1) {

      // num subpatches per patch - this is dynamic per patch, as on the regrid before
      // we could have combined several patches together
      vector<int> nsppp(l->numPatches());
      vector<int> recvcounts(d_myworld->size(),0);
      for (Level::const_patchIterator iter = l->patchesBegin(); iter != l->patchesEnd(); iter++) {
        const Patch* patch = *iter;
        IntVector patchRefinement = 
          ((patch->getCellHighIndex() - patch->getCellLowIndex()) / d_patchSize[i]) * 
          d_latticeRefinementRatio[i];
        int nsp = patchRefinement.x() * patchRefinement.y() * patchRefinement.z();
        nsppp[(*iter)->getLevelIndex()] = nsp;
        recvcounts[lb->getPatchwiseProcessorAssignment(patch)] += nsp;
      }
      
      vector<int> displs(d_myworld->size(),0);
      
      for (int p = 1; p < (int)displs.size(); p++) {
        displs[p] = displs[p-1]+recvcounts[p-1];
      }
    
      // create the buffers to send the data
      vector<int> sendbuf(recvcounts[d_myworld->myrank()]);

      int sendbufindex = 0;

      // use this to sort the allSubPatchBuffer in order of processor.  The displs
      // array keeps track of the index of the final array that holds the first
      // item of each processor.  So use this to put allSubPatchBuffer in the right
      // order, by referencing it, and then the next item for that processor will
      // go in that value + nsppp.
      vector<int> subpatchSorter = displs;
      
      // for this mpi communication, we're going to Gather all subpatch information to allprocessors.
      // We're going to get what data we have from the DW and put its pointer data in sendbuf.  While
      // we're doing that, we need to prepare the receiving buffer to receive these, so we create a subPatchFlag
      // for each patch we're going to receive from, and then put it in the right order, as the 
      // MPI_Allgatherv puts them in order of processor.
      
      for (Level::const_patchIterator citer = l->patchesBegin();
           citer != l->patchesEnd(); citer++) {
        Patch *patch = *citer;

        // the subpatchSorter's index is in terms of the numbers of the subpatches
        int index = subpatchSorter[lb->getPatchwiseProcessorAssignment(patch)];
        int nsp = nsppp[patch->getLevelIndex()];
        int proc = lb->getPatchwiseProcessorAssignment(patch);

        IntVector patchRefinement =
          ((patch->getCellHighIndex() - patch->getCellLowIndex()) / d_patchSize[i]) *
          d_latticeRefinementRatio[i];


        subpatchSorter[proc] += nsp;
        IntVector patchIndex = patch->getCellLowIndex()/d_patchSize[i];

        // create the recv buffers to put the data in
        // recvbuf ensure that the data be received in a contiguous array, and allSubpatches will
        // index into it
        SubPatchFlagP spf1 = scinew SubPatchFlag;
        
        spf1->initialize(d_latticeRefinementRatio[i]*patchIndex,
                         d_latticeRefinementRatio[i]*patchIndex+patchRefinement, &recvbuf[index]);

        allSubpatches[patch->getLevelIndex()] = spf1;

        if (proc != d_myworld->myrank())
          continue;
        
        // get the variable and prepare to send it: put its base pointer in the send buffer
        PerPatch<SubPatchFlagP> spf2;
        dw->get(spf2, d_activePatchesLabel, 0, patch);
        for (int idx = 0; idx < nsppp[patch->getLevelIndex()]; idx++) {
          sendbuf[idx + sendbufindex] = spf2.get()->subpatches_[idx];
        }
        sendbufindex += nsp;
      }

      // if we are going to try to use superpatches, then have proc 0 do the work, and broadcast when we're done
#if 1
      if (d_maxPatchSize[levelIdx] == d_patchSize[levelIdx]) {
#endif
        MPI_Allgatherv(&sendbuf[0], sendbufindex, MPI_INT, 
                       &recvbuf[0], &recvcounts[0], &displs[0], MPI_INT, d_myworld->getComm());
#if 1
      }
      else {
        MPI_Gatherv(&sendbuf[0], sendbufindex, MPI_INT,
                    &recvbuf[0], &recvcounts[0], &displs[0], MPI_INT, 0, d_myworld->getComm());
      }
#endif
    }
    else {
      // for the single-proc case, just put the subpatches in the same container
      for (Level::const_patchIterator citer = l->patchesBegin();
           citer != l->patchesEnd(); citer++) {
        Patch *patch = *citer;
        PerPatch<SubPatchFlagP> spf;
        dw->get(spf, d_activePatchesLabel, 0, patch);
        allSubpatches[patch->getLevelIndex()] = spf.get();
      }
      rdbg << "   Got data\n";

    }
#if 1
    if (d_maxPatchSize[levelIdx] == d_patchSize[levelIdx] || d_myworld->myrank() == 0) {
#endif
      // loop over each patch's subpatches (these will be the patches on level+1)
      IntVector periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
      for (unsigned j = 0; j < allSubpatches.size(); j++) {
        rdbg << "   Doing subpatches index " << j << endl;
        for (CellIterator iter(allSubpatches[j].get_rep()->low_, allSubpatches[j].get_rep()->high_); 
             !iter.done(); iter++) {
          IntVector idx(*iter);
          
          // if that subpatch is active...
          if ((*allSubpatches[j].get_rep())[idx]) {
            // add this subpatch to become the next level's patch
            rdbg << d_myworld->myrank() << " Adding normal subpatch " << idx << " to level " << i+1 << endl;
            d_patches[i+1].insert(idx);
          }
          else {
            //           rdbg << " NOT adding subpatch " << idx << " to level " << i+1 << endl;
          }
        }
      }
      // do patch dilation here instead of in above loop so we can dilate the upper level's dilation...
      if (i > 0) { // don't dilate onto level 0
        IntVector range = Ceil(d_minBoundaryCells.asVector()/d_patchSize[i].asVector());
        for (subpatchset::iterator iter = d_patches[i+1].begin(); iter != d_patches[i+1].end(); iter++) {
          IntVector idx(*iter);
          for (CellIterator inner(IntVector(-1,-1,-1)*range, range+IntVector(1,1,1)); !inner.done(); inner++) {
            // "dilate" each subpatch, adding it to the patches on the coarser level
            IntVector dilate_idx = (idx + *inner) / d_latticeRefinementRatio[i];
            // we need to wrap around for periodic boundaries
            if (((dilate_idx.x() < 0 || dilate_idx.x() >= d_patchNum[i].x()) && !periodic.x()) ||
                ((dilate_idx.y() < 0 || dilate_idx.y() >= d_patchNum[i].y()) && !periodic.y()) ||
                ((dilate_idx.z() < 0 || dilate_idx.z() >= d_patchNum[i].z()) && !periodic.z()))
              continue;
            
            // if it was periodic, get it in the proper range
            for (int d = 0; d < 3; d++) {
              while (dilate_idx[d] < 0) dilate_idx[d] += d_patchNum[i][d];
              while (dilate_idx[d] >= d_patchNum[i][d]) dilate_idx[d] -= d_patchNum[i][d];
            }
            
            rdbg << "  Adding dilated subpatch " << dilate_idx << endl;
            d_patches[i].insert(dilate_idx);
          } // end if allSubpatches[j][idx]
        } // end for celliterator
      } // end for unsigned j
#if 1
    } // end if (d_maxPatchSize[levelIdx] == d_patchSize[levelIdx] || d_myworld->myrank() == 0)
#endif
  } // end for i = toplevel

  // put level 0's patches into the set so we can just iterate through the whole set later
  for (CellIterator iter(IntVector(0,0,0), d_patchNum[0]); !iter.done(); iter++) {
    IntVector idx(*iter);
    rdbg << "Adding patch " << idx << " to level 0" << endl;
    d_patches[0].insert(idx);
  }
  rdbg << "GatherSubPatches END\n";
}

class PatchShell
{
public:
  PatchShell(IntVector low, IntVector high, IntVector in_low, IntVector in_high) :
    low(low), high(high), in_low(in_low), in_high(in_high) {}
  PatchShell() { low = high = IntVector(-9,-9,-9); }
  IntVector low, high, in_low, in_high;

  struct Compare
  {
    bool operator()(PatchShell p1, PatchShell p2) const
    { return p1.low < p2.low; }
  };
};



Grid* HierarchicalRegridder::CreateGrid2(Grid* oldGrid) 
{
  rdbg << "CreateGrid2 BGN\n";

  Grid* newGrid = scinew Grid();

  for (int levelIdx=0; levelIdx < (int)d_patches.size(); levelIdx++) {
    if (
#if 1
        (d_maxPatchSize[levelIdx] == d_patchSize[levelIdx] || d_myworld->myrank() == 0) && 
#endif
        d_patches[levelIdx].size() == 0)
      break;

    Point anchor;
    Vector spacing;
    IntVector extraCells;

    if (levelIdx < oldGrid->numLevels()) {
      anchor = oldGrid->getLevel(levelIdx)->getAnchor();
      spacing = oldGrid->getLevel(levelIdx)->dCell();
      extraCells = oldGrid->getLevel(levelIdx)->getExtraCells();
    } else {
      anchor = newGrid->getLevel(levelIdx-1)->getAnchor();
      spacing = newGrid->getLevel(levelIdx-1)->dCell() / d_cellRefinementRatio[levelIdx-1];
      extraCells = newGrid->getLevel(levelIdx-1)->getExtraCells();
    }

    LevelP newLevel = newGrid->addLevel(anchor, spacing);
    newLevel->setExtraCells(extraCells);

    rdbg << "HierarchicalRegridder::regrid(): Setting extra cells to be: " << extraCells << endl;

    // now create the patches.  Do up to 2 passes.
    // 1) if the input file specifies a patches_to_combine greater than 1,1,1, then attempt to
    // create a superpatch configuration.
    // 2) if not, just add it to the normal grid.  If so put all the super patches into the grid

    // if we do the two passes, then we need to have one processor do the work and broadcast it, as 
    // the superpatch functionality does not produce consistent results across processors

    Grid bogusGrid;
    Level* addToLevel = newLevel.get_rep();
#if 1
    if (d_maxPatchSize[levelIdx] == d_patchSize[levelIdx] || d_myworld->myrank() == 0) {
#endif
      if (d_maxPatchSize[levelIdx] != d_patchSize[levelIdx]) {
        // attempt to combine the patches together so we don't have so many patches (weakness of this regridder)
        // create a superBoxSet, and use "patches", as they have all the functionality we need
        addToLevel = bogusGrid.addLevel(Point(0,0,0), Vector(1,1,1), -999);
      }
      
      int id = -999999;
      for (subpatchset::iterator iter = d_patches[levelIdx].begin(); iter != d_patches[levelIdx].end(); iter++) {
        IntVector idx(*iter);
        rdbg << d_myworld->myrank() << "   Creating patch "<< *iter << endl;
        IntVector startCell       = idx * d_patchSize[levelIdx];
        IntVector endCell         = (idx + IntVector(1,1,1)) * d_patchSize[levelIdx] - IntVector(1,1,1);
        IntVector inStartCell(startCell);
        IntVector inEndCell(endCell);
        
        // do extra cells - add if there is no neighboring patch
        IntVector low(0,0,0), high(0,0,0);
        if (d_patches[levelIdx].find(idx-IntVector(1,0,0)) == d_patches[levelIdx].end())
          low[0] = extraCells.x();
        if (d_patches[levelIdx].find(idx-IntVector(0,1,0)) == d_patches[levelIdx].end())
          low[1] = extraCells.y();
        if (d_patches[levelIdx].find(idx-IntVector(0,0,1)) == d_patches[levelIdx].end())
          low[2] = extraCells.z();
        
        if (d_patches[levelIdx].find(idx+IntVector(1,0,0)) == d_patches[levelIdx].end())
          high[0] = extraCells.x();
        if (d_patches[levelIdx].find(idx+IntVector(0,1,0)) == d_patches[levelIdx].end())
          high[1] = extraCells.y();
        if (d_patches[levelIdx].find(idx+IntVector(0,0,1)) == d_patches[levelIdx].end())
          high[2] = extraCells.z();
        
        if (idx.x() == d_patchNum[levelIdx](0)-1) endCell(0) = d_cellNum[levelIdx](0)-1;
        if (idx.y() == d_patchNum[levelIdx](1)-1) endCell(1) = d_cellNum[levelIdx](1)-1;
        if (idx.z() == d_patchNum[levelIdx](2)-1) endCell(2) = d_cellNum[levelIdx](2)-1;
        
        rdbg << "     Adding extra cells: " << low << ", " << high << endl;
        
        startCell -= low;
        endCell += high;
        
        /// pass in our own id to not increment the global id
        int patchID = (d_maxPatchSize[levelIdx] != d_patchSize[levelIdx]) ? id++ : -1;
        addToLevel->addPatch(startCell, endCell + IntVector(1,1,1), 
			     inStartCell, inEndCell + IntVector(1,1,1), newGrid, patchID);
      }
#if 1
    }
#endif

    // do the second pass if we did the superpatch pass
    if (d_maxPatchSize[levelIdx] != d_patchSize[levelIdx]) {
      int size;
      vector<PatchShell> finalPatches;
#if 1
      if (d_myworld->myrank() == 0) {
#endif
        const SuperPatchContainer* superPatches;
        LocallyComputedPatchVarMap patchGrouper;
        const PatchSubset* patches = addToLevel->allPatches()->getUnion();
        patchGrouper.addComputedPatchSet(patches);
        patchGrouper.makeGroups();
        superPatches = patchGrouper.getSuperPatches(addToLevel);

        SuperPatchContainer::const_iterator iter;
        for (iter = superPatches->begin(); iter != superPatches->end(); iter++) {
          const SuperPatch* superBox = *iter;
          const Patch* firstPatch = superBox->getBoxes()[0];
          IntVector low(firstPatch->getExtraCellLowIndex()), high(firstPatch->getExtraCellHighIndex());
          IntVector in_low(firstPatch->getCellLowIndex()), in_high(firstPatch->getCellHighIndex());
          for (unsigned int i = 1; i < superBox->getBoxes().size(); i++) {
            // get the minimum extents containing both the expected ghost cells
            // to be needed and the given ghost cells.
            const Patch* memberPatch = superBox->getBoxes()[i];
            low = Min(memberPatch->getExtraCellLowIndex(), low);
            high = Max(memberPatch->getExtraCellHighIndex(), high);
            in_low = Min(memberPatch->getCellLowIndex(), in_low);
            in_high = Max(memberPatch->getCellHighIndex(), in_high);
          }
          finalPatches.push_back(PatchShell(low, high, in_low, in_high));
          //cout << d_myworld->myrank() << "  Adding " << low << endl;
        }
        
#if 1
        // sort the superboxes.  On different iterations, the same patches can be sorted
        // differently, and thus force the regrid to happen when normally we would do nothing
        //PatchShell::Compare compare;
        //sort(finalPatches.begin(), finalPatches.end(), compare);
        size = finalPatches.size();
      }

      if (d_myworld->size() > 1) {
        MPI_Bcast(&size, 1, MPI_INT, 0, d_myworld->getComm());
        finalPatches.resize(size);
        MPI_Bcast(&finalPatches[0], size*4*3, MPI_INT, 0, d_myworld->getComm());
      }
#endif
      for (unsigned i = 0; i < finalPatches.size(); i++) {
        PatchShell& ps = finalPatches[i];
        IntVector low(ps.low), high(ps.high), in_low(ps.in_low), in_high(ps.in_high);
        // split up the superpatch into pieces no bigger than maxPatchSize
        IntVector divisions(Ceil((in_high-in_low).asVector() / d_maxPatchSize[levelIdx]));
        rdbg  << "  superpatch needs " << divisions << " divisions " << endl;
        for (int x = 0; x < divisions.x(); x++) {
          for (int y = 0; y < divisions.y(); y++) {
            for (int z = 0; z < divisions.z(); z++) {
              IntVector idx(x,y,z);

              // we must align the combined superpatches with the lattice, so don't split it into even pieces (in cases of rounding), 
              // but according to original patch structure
              IntVector sub_in_low = in_low + d_maxPatchSize[levelIdx]*idx;
              IntVector sub_in_high = Min(in_high, in_low + d_maxPatchSize[levelIdx] * (idx+IntVector(1,1,1)));

              IntVector sub_low = sub_in_low, sub_high = sub_in_high;
              for (int dim = 0; dim < 3; dim++) {
                if (idx[dim] == 0) sub_low[dim] = low[dim];
                if (idx[dim] == divisions[dim] - 1) sub_high[dim] = high[dim];
              }
              // finally add the superpatch to the real level
              if (d_myworld->myrank() == 0)
                rdbg << "   Using superpatch " << sub_low << " " << sub_high << " " << sub_in_low << " " << sub_in_high << endl;
              newLevel->addPatch(sub_low, sub_high, sub_in_low, sub_in_high,newGrid);
            }
          }
        }
      }
    }
  }

  d_newGrid = true;
  d_lastRegridTimestep = d_sharedState->getCurrentTopLevelTimeStep();

  rdbg << "HierarchicalRegridder::regrid() END" << endl;

  rdbg << "CreateGrid2 END\n";
  if (*newGrid == *oldGrid) {
    d_newGrid = false;
    delete newGrid;
    return oldGrid;
  }
  
  // do this after the grid check, as it's expensive and we don't want to do it if we're just going to throw it away
  for (int levelIdx = 0; levelIdx < newGrid->numLevels(); levelIdx++) {
    LevelP lev = newGrid->getLevel(levelIdx);
    IntVector periodic;
    if(levelIdx == 0){
      periodic = oldGrid->getLevel(0)->getPeriodicBoundaries();
    } else {
      periodic = newGrid->getLevel(levelIdx-1)->getPeriodicBoundaries();
    }
    lev->finalizeLevel(periodic.x(), periodic.y(), periodic.z());
    //lev->assignBCS(grid_ps_,0);
  }

  newGrid->performConsistencyCheck();
  return newGrid;
}

IntVector HierarchicalRegridder::calculateNumberOfPatches(IntVector& cellNum, IntVector& patchSize)
{
  IntVector patchNum = Ceil(Vector(cellNum.x(), cellNum.y(), cellNum.z()) / patchSize);
  IntVector remainder = Mod(cellNum, patchSize);

  if (remainder.x() || remainder.y() || remainder.z()) {
    ostringstream msg;
    msg << "  HierarchicalRegridder: The domain (" << cellNum[0] << "x" << cellNum[1] << "x" << cellNum[2] 
        << " cells) is not divisible by the number of patches (" << patchSize[0] << "x" << patchSize[1] << "x" << patchSize[2] 
        << " patches)\n";
    throw ProblemSetupException(msg.str(), __FILE__, __LINE__);
  }
  
  return patchNum;
}


