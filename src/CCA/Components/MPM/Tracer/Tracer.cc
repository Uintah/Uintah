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
#include <CCA/Components/MPM/Tracer/Tracer.h>
#include <CCA/Components/MPM/Tracer/TracerMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <fstream>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
Tracer::Tracer(TracerMaterial* tm, MPMFlags* flags,
                           SimulationStateP& ss)
{
  d_lb = scinew MPMLabel();

  d_flags = flags;

  d_sharedState = ss;

  registerPermanentTracerState(tm);
}

Tracer::~Tracer()
{
  delete d_lb;
}
//______________________________________________________________________
ParticleSubset* 
Tracer::createTracers(TracerMaterial* matl,
                      particleIndex numTracers,
                      CCVariable<short int>& cellNAPID,
                      const Patch* patch,
                      DataWarehouse* new_dw,
                      const string filename)
{
  int dwi = matl->getDWIndex();
  ParticleSubset* subset = allocateVariables(numTracers,dwi,patch,new_dw);

  particleIndex start = 0;

  if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening tracer file "+filename+" in createTracers \n",
                                  __FILE__, __LINE__);
    }

    // Field for position
    double p1,p2,p3;
    int line = 0;
    while(is >> p1 >> p2 >> p3){
      Point pos = Point(p1,p2,p3);
      line++;
//      IntVector cell_idx;
      if(patch->containsPoint(pos)){
        particleIndex pidx = start;
        tracer_pos[pidx]   = pos;
        // Use the tracer's original line # in the .pts file as its unique ID
        tracerID[pidx] = line;
        start++;

//        // Figure out unique ID for the Tracer
//        patch->findCell(pos,cell_idx);
//        ASSERT(cell_idx.x() <= 0xffff &&
//               cell_idx.y() <= 0xffff &&
//               cell_idx.z() <= 0xffff);

//        long64 cellID = ((long64)cell_idx.x() << 16) |
//                        ((long64)cell_idx.y() << 32) |
//                        ((long64)cell_idx.z() << 48);

//        short int& myCellNAPID = cellNAPID[cell_idx];
//        tracerID[pidx] = (cellID | (long64) myCellNAPID);
//        ASSERT(myCellNAPID < 0x7fff);
//        myCellNAPID++;
      }
    }  // while
    is.close();
  }

  return subset;
}

//__________________________________
//
ParticleSubset* 
Tracer::allocateVariables(particleIndex numTracers, 
                                int dwi, const Patch* patch,
                                DataWarehouse* new_dw)
{

  ParticleSubset* subset = new_dw->createParticleSubset(numTracers,dwi,patch);

  new_dw->allocateAndPut(tracer_pos,     d_lb->pXLabel,             subset);
  new_dw->allocateAndPut(tracerID,       d_lb->tracerIDLabel,       subset);
  
  return subset;
}

//__________________________________
//
particleIndex 
Tracer::countTracers(const Patch* patch, const string filename)
{
  particleIndex sum = 0;

  if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening cohesive zone file "+filename+" in countTracers\n",
                                  __FILE__, __LINE__);
    }

    // Field for position, normal, tangential and length.
    // Everything else is assumed to be zero.
    double f1,f2,f3;
    while(is >> f1 >> f2 >> f3){
      if(patch->containsPoint(Point(f1,f2,f3))){
        sum++;
      } else {
      }
    }
    is.close();
  }

  return sum;
}
//__________________________________
//
vector<const VarLabel* > Tracer::returnTracerState()
{
  return d_tracer_state;
}
//__________________________________
//
vector<const VarLabel* > Tracer::returnTracerStatePreReloc()
{
  return d_tracer_state_preReloc;
}
//__________________________________
//
void Tracer::registerPermanentTracerState(TracerMaterial* trmat)
{
  d_tracer_state.push_back(d_lb->tracerIDLabel);
  d_tracer_state_preReloc.push_back(d_lb->tracerIDLabel_preReloc);
}
//__________________________________
//
void Tracer::scheduleInitialize(const LevelP& level, 
                                      SchedulerP& sched)
{
  Task* t = scinew Task("Tracer::initialize",
                  this, &Tracer::initialize);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(d_lb->pXLabel);
  t->computes(d_lb->tracerIDLabel);
  t->computes(d_lb->pCellNATracerIDLabel,zeroth_matl);

  sched->addTask(t, level->eachPatch(), d_sharedState->allTracerMaterials());

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...
}

//__________________________________
//
void Tracer::initialize(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* tracer_matls,
                              DataWarehouse* ,
                              DataWarehouse* new_dw)
{
  particleIndex totalTracers=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

//  printTask(patches, patch,cout_doing,"Doing initialize for Tracers\t");

    CCVariable<short int> cellNATracerID;
    new_dw->allocateAndPut(cellNATracerID, d_lb->pCellNATracerIDLabel, 0,patch);
    cellNATracerID.initialize(0);

    for(int m=0;m<tracer_matls->size();m++){
      TracerMaterial* tracer_matl = d_sharedState->getTracerMaterial( m );
      string filename = tracer_matl->getTracerFilename();
      particleIndex numTracers = countTracers(patch,filename);
      totalTracers+=numTracers;

      createTracers(tracer_matl, numTracers, cellNATracerID,
                    patch, new_dw, filename);
    }
//    cout << "Total Tracers " << totalTracers << endl;
  }
}
