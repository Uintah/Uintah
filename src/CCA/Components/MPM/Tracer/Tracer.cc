/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
#include <CCA/Components/MPM/Core/TracerLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <fstream>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
Tracer::Tracer(TracerMaterial* tm, MPMFlags* flags,
                           MaterialManagerP& ss)
{
  d_lb = scinew MPMLabel();
  d_TL = scinew TracerLabel();

  d_flags = flags;

  d_materialManager = ss;

  registerPermanentTracerState(tm);
}

Tracer::~Tracer()
{
  delete d_lb;
  delete d_TL;
}
//______________________________________________________________________
ParticleSubset* 
Tracer::createTracers(TracerMaterial* matl,
                      particleIndex numTracers,
                      const Patch* patch,
                      DataWarehouse* new_dw,
                      const string filename)
{
  int dwi = matl->getDWIndex();
  ParticleSubset* subset = allocateVariables(numTracers,dwi,patch,new_dw);


  if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening tracer file "+filename+" in createTracers \n",
                                  __FILE__, __LINE__);
    }

    double p1,p2,p3;
    string line;
    particleIndex start = 0;
    while (getline(is, line)) {
     istringstream ss(line);
     string token;
     long64 tid;
     ss >> token;
     tid = stoull(token);
     ss >> token;
     p1 = stof(token);
     ss >> token;
     p2 = stof(token);
     ss >> token;
     p3 = stof(token);
//     cout << tid << " " << p1 << " " << p2 << " " << p3 << endl;
     Point pos = Point(p1,p2,p3);
     if(patch->containsPoint(pos)){
       particleIndex pidx = start;
       tracer_pos[pidx]   = pos;
       tracerID[pidx] = tid;
       start++;
     }
    }
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
  new_dw->allocateAndPut(tracerID,       d_TL->tracerIDLabel,       subset);
  
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
      throw ProblemSetupException("ERROR Opening tracer file "+filename+" in countTracers\n",
                                  __FILE__, __LINE__);
    }

    string line;

    while (getline(is, line)) {
     istringstream ss(line);
     string token;
     long64 tid;
     double f1,f2,f3;
     ss >> token;
     tid = stoull(token);
     ss >> token;
     f1 = stof(token);
     ss >> token;
     f2 = stof(token);
     ss >> token;
     f3 = stof(token);
     if(patch->containsPoint(Point(f1,f2,f3))){
//       cout << tid << " " << f1 << " " << f2 << " " << f3 << endl;
       sum++;
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
  d_tracer_state.push_back(d_TL->tracerIDLabel);
  d_tracer_state_preReloc.push_back(d_TL->tracerIDLabel_preReloc);
}
//__________________________________
//
void Tracer::scheduleInitialize(const LevelP& level,
                                      SchedulerP& sched, MaterialManagerP &mm)
{
  Task* t = scinew Task("Tracer::initialize",
                  this, &Tracer::initialize);

  t->computes(d_lb->pXLabel);
  t->computes(d_TL->tracerIDLabel);
  t->computes(d_TL->tracerCountLabel);

  sched->addTask(t, level->eachPatch(), mm->allMaterials("Tracer"));
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

    //printTask(patches, patch,cout_doing,"Doing initialize for Tracers\t");

    for(int m=0;m<tracer_matls->size();m++){
      TracerMaterial* tracer_matl = 
                  (TracerMaterial*) d_materialManager->getMaterial("Tracer", m);
      string filename = tracer_matl->getTracerFilename();
      particleIndex numTracers = countTracers(patch,filename);
      totalTracers+=numTracers;

      createTracers(tracer_matl, numTracers,
                    patch, new_dw, filename);
    }
    new_dw->put(sumlong_vartype(totalTracers), d_TL->tracerCountLabel);
    //cout << "Total Tracers " << totalTracers << endl;
  }
}
