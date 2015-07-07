/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 *
 * ----------------------------------------------------------
 * RetrievalTest.cc
 *
 *  Created on: Apr 3, 2015
 *      Author: jbhooper
 */

#include <CCA/Components/Examples/RetrievalTest.h>

#include <CCA/Ports/Scheduler.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Datatypes/String.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Mutex.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace Uintah;

SCIRun::Mutex outputLock("file output lock");

patchData::patchData(const SCIRun::IntVector centerLow,
                     const SCIRun::IntVector centerHigh,
                     const Patch* current,
                     const int neighborsSeen, const int neighborsExpected)
                    :d_patchReference(current),
                     d_neighboringCells(neighborsSeen),
                     d_expectedCells(neighborsExpected)
{
  SCIRun::IntVector patchHigh   = current->getCellHighIndex();
  SCIRun::IntVector patchLow    = current->getCellLowIndex();
  SCIRun::IntVector patchSize   = patchHigh - patchLow;

  SCIRun::IntVector centerSize  = centerHigh - centerLow;

  SCIRun::IntVector numLayers = (patchHigh + patchLow - centerHigh - centerLow)/
                                (centerSize + patchSize);

  d_layer = Max(abs(numLayers.x()),Max(abs(numLayers.y()),abs(numLayers.z())));

  // We can also determine if the patch is "face-like", "edge-like", or "corner-like"
  // for the level by comparing the layer to the components of the layer vectors.
  d_center[0] = (patchHigh[0]+patchLow[0])/2.0;
  d_center[1] = (patchHigh[1]+patchLow[1])/2.0;
  d_center[2] = (patchHigh[2]+patchLow[2])/2.0;
  int layerCount = 0;
  for (int index=0; index < 3; ++index)
  {
    if (abs(numLayers[index]) == d_layer)
    {
      ++layerCount;
    }
  }
  switch (layerCount)
  {
    case (0):
        d_patchType = patchData::interior;
        break;
    case (1):
        d_patchType = patchData::face;
        break;
    case (2):
        d_patchType = patchData::edge;
        break;
    case (3):
        d_patchType = patchData::corner;
        break;
  }
  if (d_layer == 0) d_patchType = patchData::interior;
}



std::ostream& Uintah::operator<<(std::ostream& osOut, const Uintah::patchData& data)
{
  std::ios_base::fmtflags OSFlags=osOut.flags();
  osOut << " Patch: " << std::setw(4) << std::right << data.ID()
        << " - Center ["
        << std::setw(6) << std::right << std::fixed << std::setprecision(2)
        << data.x()
        << std::setw(6) << std::right << std::fixed << std::setprecision(2)
        << data.y()
        << std::setw(6) << std::right << std::fixed << std::setprecision(2)
        << data.z() << "] | ";
  switch (data.d_patchType)
  {
    case (patchData::interior):
      osOut << " Interior Cell -> Layer ";
      break;
    case (patchData::face):
      osOut << "     Face Cell -> Layer ";
      break;
    case (patchData::edge):
      osOut << "     Edge Cell -> Layer ";
      break;
    case (patchData::corner):
      osOut << "   Corner Cell -> Layer ";
      break;
  }
  osOut << std::setw(3) << std::right << data.d_layer << " ==>"
        << std::setw(5) << std::right << data.getNeighborCells() << "/"
        << std::setw(5) << std::left  << data.getExpectedCells()
        << " cells returned by neighbor request.";
  osOut.flags(OSFlags);
  return osOut;
}

bool patchData::operator<(const patchData& rhs) const
{
  // Sort order: X, Y, Z
  if (d_center[0] > rhs.d_center[0]) return false;
  if (d_center[0] == rhs.d_center[0] && d_center[1] > rhs.d_center[1]) return false;
  if (d_center[1]== rhs.d_center[1] && d_center[2] > rhs.d_center[2]) return false;
  return true;
}

RetrievalTest::RetrievalTest(const ProcessorGroup* myWorld)
                            : UintahParallelComponent(myWorld)
{
  pX            = VarLabel::create("p.x",
                                   ParticleVariable<Point>::getTypeDescription());
  pX_preReloc   = VarLabel::create("p.x+",
                                   ParticleVariable<Point>::getTypeDescription());
  pID           = VarLabel::create("p.ID",
                                   ParticleVariable<long64>::getTypeDescription());
  pID_preReloc  = VarLabel::create("p.ID+",
                                   ParticleVariable<long64>::getTypeDescription());
  pLocation     = VarLabel::create("p.loc",
                                   ParticleVariable<SCIRun::Vector>::getTypeDescription());
  pLocation_preReloc = VarLabel::create("p.loc+",
                                        ParticleVariable<SCIRun::Vector>::getTypeDescription());

  d_CellsPerPatch = 0;

}

RetrievalTest::~RetrievalTest()
{
  VarLabel::destroy(pX);
  VarLabel::destroy(pX_preReloc);
  VarLabel::destroy(pID);
  VarLabel::destroy(pID_preReloc);
  VarLabel::destroy(pLocation);
  VarLabel::destroy(pLocation_preReloc);
}

void RetrievalTest::problemSetup(const ProblemSpecP&     problemSpec,
                                 const ProblemSpecP&     restartSpec,
                                       GridP&            grid,
                                       SimulationStateP& sharedState)
{
  d_sharedState = sharedState;
  dynamic_cast<Scheduler*> (getPort("scheduler"))->setPositionVar(pX);

  ProblemSpecP retrieval_ps = problemSpec->findBlock("RetrievalTest");
  retrieval_ps->get("numGhostCells",d_numGhostCells);
  retrieval_ps->get("outputFile",d_outfileName);

  // Determine the total number of patches in the system
  int num_Patches = grid->getLevel(0)->numPatches();
  // Number of cells in the system
  int num_Cells = grid->getLevel(0)->totalCells();

  d_CellsPerPatch = static_cast<int> (std::ceil
                                      (static_cast<double> (num_Cells) /
                                        static_cast<double> (num_Patches) )
                                     );

  referenceMaterial = scinew SimpleMaterial();
  d_sharedState->registerSimpleMaterial(referenceMaterial);
  registerParticleState();

  // Store the center cell's high/low indices across all processors for building
  // data output objects.
  SCIRun::IntVector centerCell(-1,-1,-1);
  SCIRun::IntVector cellLow, cellHigh;
  grid->getLevel(0)->findInteriorCellIndexRange(cellLow,cellHigh);
  centerCell = (cellHigh + cellLow)/SCIRun::IntVector(2,2,2);
  const Patch* centerPatch = grid->getLevel(0)->getPatchFromIndex(centerCell,false);
  d_centerHigh = centerPatch->getCellHighIndex();
  d_centerLow  = centerPatch->getCellLowIndex();

  std::ofstream particleNeighborOut;
  particleNeighborOut.open(d_outfileName.c_str());
  particleNeighborOut.close();
}

void RetrievalTest::scheduleRestartInitialize(const LevelP&     level,
                                                    SchedulerP& sched)
{

}

void RetrievalTest::scheduleInitialize(const LevelP&        level,
                                             SchedulerP&    sched)
{
  Task* task = scinew Task("initialize",
                           this,
                           &RetrievalTest::initialize);

  task->computes(pX);
  task->computes(pID);
  task->computes(pLocation);

  task->setType(Task::OncePerProc);

   LoadBalancer* loadBal = sched->getLoadBalancer();
   const PatchSet* perProcPatches = loadBal->getPerProcessorPatchSet(level);

  sched->addTask(task, perProcPatches, d_sharedState->allMaterials());

}

void RetrievalTest::scheduleComputeStableTimestep(const LevelP&     level,
                                                        SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep",
                           this,
                           &RetrievalTest::computeStableTimestep);
  // Something here to gate this step on other results?
  task->computes(d_sharedState->get_delt_label(), level.get_rep());
  sched->addTask(task,level->eachPatch(), d_sharedState->allMaterials());
}

void RetrievalTest::scheduleTimeAdvance(const LevelP&       level,
                                              SchedulerP&   sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* materials = d_sharedState->allMaterials();

  scheduleRetrieveData(sched, patches, materials);

  sched->scheduleParticleRelocation(level,
                                    pX_preReloc, d_sharedState->d_particleState_preReloc,
                                    pX, d_sharedState->d_particleState,
                                    pID,
                                    materials);
}

void RetrievalTest::scheduleRetrieveData(      SchedulerP&  sched,
                                         const PatchSet*    patches,
                                         const MaterialSet* materials)
{
  Task* task = scinew Task("retrieveData",
                           this,
                           &RetrievalTest::retrieveData);

  // Tasks which require a fixed number of cells
  task->requires(Task::OldDW, pX,        Ghost::AroundCells, d_numGhostCells);
  task->requires(Task::OldDW, pID,       Ghost::AroundCells, d_numGhostCells);
  task->requires(Task::OldDW, pLocation, Ghost::AroundCells, d_numGhostCells);

  task->computes(pX_preReloc);
  task->computes(pID_preReloc);
  task->computes(pLocation_preReloc);

  sched->addTask(task, patches, materials);
}

void RetrievalTest::initialize(const ProcessorGroup* pg,
                               const PatchSubset*    patches,
                               const MaterialSubset* materials,
                                     DataWarehouse* /*oldDW*/,
                                     DataWarehouse*  newDW)
{

  unsigned int numPatches = patches->size();

  SCIRun::IntVector centerCell(-1,-1,-1);
  if (numPatches > 0)
  {
    const Patch* currPatch = patches->get(0);
    SCIRun::IntVector low, high;
    currPatch->getLevel()->findInteriorCellIndexRange(low,high);
    centerCell = (high + low)/SCIRun::IntVector(2);
  }

  for (unsigned int currPatch = 0; currPatch < numPatches; ++currPatch)
  {
    const Patch* workPatch = patches->get(currPatch);

    // Create one particle per cell
    ParticleSubset* pset = newDW->createParticleSubset((workPatch->getCellIterator()).size(),
                                                       referenceMaterial->getDWIndex(),
                                                       workPatch);
    ParticleVariable<SCIRun::Point> particlePosition;
    ParticleVariable<long64> particleID;
    ParticleVariable<SCIRun::Vector> particleLabel;

    newDW->allocateAndPut(particlePosition, pX, pset);
    newDW->allocateAndPut(particleID, pID, pset);
    newDW->allocateAndPut(particleLabel, pLocation, pset);

    int localParticleIndex = 0; // Needed to index into the particle subset
    int globalParticleOffset = d_CellsPerPatch * workPatch->getID();

    for (CellIterator currCell=workPatch->getCellIterator();
                     !currCell.done();
                   ++currCell)
    {
      SCIRun::Point pPosition = (workPatch->getCellPosition(*currCell));
      particleLabel[localParticleIndex]     = pPosition.asVector();
      particlePosition[localParticleIndex]  = pPosition;
      particleID[localParticleIndex]        = globalParticleOffset
                                             + localParticleIndex;
      ++localParticleIndex;
    }

  }
}

void RetrievalTest::computeStableTimestep(const ProcessorGroup* pg,
                                          const PatchSubset*    patches,
                                          const MaterialSubset* materials,
                                                DataWarehouse*  oldDW,
                                                DataWarehouse*  newDW)
{
  newDW->put(delt_vartype(1), d_sharedState->get_delt_label(), getLevel(patches));
}

void RetrievalTest::retrieveData(const ProcessorGroup* pg,
                                 const PatchSubset*    patches,
                                 const MaterialSubset* materials,
                                       DataWarehouse*  oldDW,
                                       DataWarehouse*  newDW)
{
  int numPatches = patches->size();
  int numMaterials = materials->size();

  int cellsPerSide  = (2*d_numGhostCells + pow(d_CellsPerPatch,1.0/3.0));
  int cellsExpected = pow(cellsPerSide,3.0);

//  SCIRun::IntVector centerCell(-1,-1,-1);
//  if (numPatches > 0)
//  {
//    const Patch* currPatch = patches->get(0);
//    SCIRun::IntVector low, high;
//    currPatch->getLevel()->findInteriorCellIndexRange(low,high);
//    centerCell = (high + low)/SCIRun::IntVector(2);
//  }

  std::vector<patchData> patchNeighborList;
  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);

    SCIRun::IntVector cellHigh = currPatch->getCellHighIndex();
    SCIRun::IntVector cellLow  = currPatch->getCellLowIndex();

//    SCIRun::IntVector cellRegionHigh = cellHigh + SCIRun::IntVector(d_numGhostCells);
//    SCIRun::IntVector cellRegionLow  = cellLow  - SCIRun::IntVector(d_numGhostCells);

    for (int matIndex = 0; matIndex < numMaterials; ++matIndex)
    {
      int currentMaterial = materials->get(matIndex);
      // This does nothing, but we seem to need it?
      ParticleSubset* delset = scinew ParticleSubset(0, currentMaterial, currPatch);
      newDW->deleteParticles(delset);

      // Transfer on every patch from old to new DW...
      ParticleSubset* selfSet = oldDW->getParticleSubset(currentMaterial,
                                                         currPatch,
                                                         Ghost::None,
                                                         0,
                                                         pX);
      constParticleVariable<SCIRun::Vector>  pSelfRead;
      constParticleVariable<Point>           pSelfPosition;
      constParticleVariable<long64>          pSelfIndex;

      oldDW->get(pSelfRead,     pLocation,  selfSet);
      oldDW->get(pSelfPosition, pX,         selfSet);
      oldDW->get(pSelfIndex,    pID,        selfSet);

      int numSelf = selfSet->numParticles();

      ParticleVariable<SCIRun::Vector>    pSelfWrite;
      ParticleVariable<Point>             pSelfPositionWrite;
      ParticleVariable<long64>            pSelfIndexWrite;
      // Allocate post relocate variables
      newDW->allocateAndPut(pSelfWrite,         pLocation_preReloc, selfSet);
      newDW->allocateAndPut(pSelfPositionWrite, pX_preReloc,        selfSet);
      newDW->allocateAndPut(pSelfIndexWrite,    pID_preReloc,       selfSet);
      for (int currParticle = 0; currParticle < numSelf; ++currParticle)
      {
        pSelfWrite[currParticle]    = pSelfRead[currParticle];
        pSelfPositionWrite[currParticle]= pSelfPosition[currParticle];
        pSelfIndexWrite[currParticle]   = pSelfIndex[currParticle];
      }

      // We only want to look at neighbors of the specific cell.
      // This is the patch we actually care about looking at.
//      if (currPatch->containsCell(centerCell))
//      {
//        std::cerr << "Center cell: [" << centerCell.x()
//                  << ", " << centerCell.y() << ", " << centerCell.z()
//                  << "]" << std::endl;
//
//
//        std::cerr << " Patch ID: " << currPatch->getID()
//                  << " Low: " << cellLow << " High: " << cellHigh << std::endl;
      std::vector<SCIRun::Vector> particleFromPset;
      ParticleSubset* neighborSet = oldDW->getParticleSubset(currentMaterial,
                                                             currPatch,
                                                             Ghost::AroundCells,
                                                             d_numGhostCells,
                                                             pX);

      constParticleVariable<SCIRun::Vector>   pNeighborRead;
      oldDW->get(pNeighborRead,   pLocation,  neighborSet);

      // Loop through set including neighbors
      int numParticles = neighborSet->numParticles();
      for (int currParticle = 0; currParticle < numParticles; ++currParticle)
      {
        particleFromPset.push_back(pNeighborRead[currParticle]);
      }
      patchNeighborList.push_back(patchData(d_centerLow,d_centerHigh,
                                            currPatch,
                                            numParticles, cellsExpected));

//      }
    }
  } // End patch loop
  // Output patch data
  std::ofstream particleNeighborOut;
  particleNeighborOut.open(d_outfileName.c_str(), std::fstream::app);

  std::sort(patchNeighborList.begin(),patchNeighborList.end());
  outputLock.lock();
  for (int index = 0; index < patchNeighborList.size(); ++index)
  {
    particleNeighborOut << patchNeighborList[index] << std::endl;
  }
  outputLock.unlock();
  particleNeighborOut.close();

}

void RetrievalTest::outputVectors(const int                             numPerLine,
                                        std::ofstream&                  outFile,
                                        std::vector<SCIRun::Vector>&    dataSet)
{
  std::vector<SCIRun::Vector>::iterator vecIt = dataSet.begin();
  int numTraversed = 0;

  while (vecIt != dataSet.end())
  {
    outFile << *vecIt << ",";
    ++vecIt;
    ++numTraversed;
    if (numTraversed == numPerLine)
    {
      outFile << std::endl;
      numTraversed = 0;
    }
  }
  if (numTraversed != 0)
  {
    outFile << std::endl;
  }

}

void RetrievalTest::outputIntVectors(const int                             numPerLine,
                                           std::ofstream&                  outFile,
                                           std::vector<SCIRun::IntVector>& dataSet)
{
  std::vector<SCIRun::IntVector>::iterator vecIt = dataSet.begin();
  int numTraversed=0;

  while (vecIt != dataSet.end())
  {
    outFile << *vecIt << ",";
    ++vecIt;
    ++numTraversed;
    if (numTraversed == numPerLine)
    {
      outFile << std::endl;
      numTraversed = 0;
    }
  }
  if (numTraversed != 0)
  {
    outFile << std::endl;
  }
  return;
}

void RetrievalTest::registerParticleState()
{
  d_particleState.push_back(pLocation);
  d_particleState.push_back(pID);

  d_particleState_preReloc.push_back(pLocation_preReloc);
  d_particleState_preReloc.push_back(pID_preReloc);

  d_sharedState->d_particleState.push_back(d_particleState);
  d_sharedState->d_particleState_preReloc.push_back(d_particleState_preReloc);
}
