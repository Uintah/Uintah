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
#include <CCA/Components/MPM/CohesiveZone/CohesiveZone.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Labels/MPMLabel.h>

#include <fstream>
#include <iostream>

using namespace Uintah;
using namespace std;

//______________________________________________________________________
//  Reference: N. P. Daphalapukar, Hongbing Lu, Demir Coker, Ranga Komanduri,
// " Simulation of dynamic crack growth using the generalized interpolation material
// point (GIMP) method," Int J. Fract, 2007, 143:79-102
//______________________________________________________________________
CohesiveZone::CohesiveZone(CZMaterial* czmat, MPMFlags* flags,
                           SimulationStateP& ss)
{
  d_lb = scinew MPMLabel();

  d_flags = flags;

  d_sharedState = ss;

  registerPermanentCohesiveZoneState(czmat);
}

CohesiveZone::~CohesiveZone()
{
  delete d_lb;
}
//______________________________________________________________________
ParticleSubset* 
CohesiveZone::createCohesiveZones(CZMaterial* matl,
                                 particleIndex numCohesiveZones,
                                 CCVariable<short int>& cellNAPID,
                                 const Patch* patch,
                                 DataWarehouse* new_dw,
                                 const string filename)
{
  int dwi = matl->getDWIndex();
  ParticleSubset* subset = allocateVariables(numCohesiveZones,dwi,patch,new_dw);

  particleIndex start = 0;

  if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening cohesive zone file "+filename+" in createCohesiveZones \n",
                                  __FILE__, __LINE__);
    }

    // needed for bulletproofing
    vector<int> mpmMatlIndex;
    int numMPM = d_sharedState->getNumMPMMatls();
    
    for(int m = 0; m < numMPM; m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      mpmMatlIndex.push_back(dwi);
    }
    
     // Field for position, normal, tangential and length.
    // Everything else is assumed to be zero.
    
    double p1,p2,p3,l4,n5,n6,n7,t8,t9,t10;
    int mt, mb;
    while(is >> p1 >> p2 >> p3 >> l4 >> n5 >> n6 >> n7 >> t8 >> t9 >> t10 >> mb >> mt){
    
      //__________________________________
      // bulletproofing
      //  the top
      int test1 = count (mpmMatlIndex.begin(), mpmMatlIndex.end(), mb);
      int test2 = count (mpmMatlIndex.begin(), mpmMatlIndex.end(), mt);
      
      if(test1 == 0 || test2 == 0 ){
        ostringstream warn;
        warn<<"ERROR:MPM:createCohesiveZones\n In the cohesive zone file ("+filename+ ") either the top/bottom material";
        warn<< "(top: " << mt << " bottom: " << mb<< ") is not a MPM material ";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
      
      Point pos = Point(p1,p2,p3);
      IntVector cell_idx;
      if(patch->containsPoint(pos)){
        particleIndex pidx = start;
        czposition[pidx]  = pos;
        czlength[pidx]    = l4;
        cznormal[pidx]    = Vector(n5,n6,n7);
        cztang[pidx]      = Vector(t8,t9,t10);
        czdisptop[pidx]   = Vector(0.0,0.0,0.0);
        czdispbottom[pidx]= Vector(0.0,0.0,0.0);
        czSeparation[pidx]= Vector(0.0,0.0,0.0);
        czForce[pidx]     = Vector(0.0,0.0,0.0);
        czBotMat[pidx]    = mb;
        czTopMat[pidx]    = mt;
        czFailed[pidx]    = 0;

        // Figure out unique ID for the CZ
        patch->findCell(pos,cell_idx);
        ASSERT(cell_idx.x() <= 0xffff &&
               cell_idx.y() <= 0xffff &&
               cell_idx.z() <= 0xffff);

        long64 cellID = ((long64)cell_idx.x() << 16) |
                        ((long64)cell_idx.y() << 32) |
                        ((long64)cell_idx.z() << 48);

        short int& myCellNAPID = cellNAPID[cell_idx];
        czID[pidx] = (cellID | (long64) myCellNAPID);
        ASSERT(myCellNAPID < 0x7fff);
        myCellNAPID++;
        start++;
      }
    }  // while
    is.close();
  }

  return subset;
}

//__________________________________
//
ParticleSubset* 
CohesiveZone::allocateVariables(particleIndex numCZs, 
                                int dwi, const Patch* patch,
                                DataWarehouse* new_dw)
{

  ParticleSubset* subset = new_dw->createParticleSubset(numCZs,dwi,patch);

  new_dw->allocateAndPut(czposition,     d_lb->pXLabel,             subset);
  new_dw->allocateAndPut(czlength,       d_lb->czLengthLabel,       subset); 
  new_dw->allocateAndPut(cznormal,       d_lb->czNormLabel,         subset);
  new_dw->allocateAndPut(cztang,         d_lb->czTangLabel,         subset);
  new_dw->allocateAndPut(czdisptop,      d_lb->czDispTopLabel,      subset);
  new_dw->allocateAndPut(czdispbottom,   d_lb->czDispBottomLabel,   subset);
  new_dw->allocateAndPut(czID,           d_lb->czIDLabel,           subset);
  new_dw->allocateAndPut(czSeparation,   d_lb->czSeparationLabel,   subset);
  new_dw->allocateAndPut(czForce,        d_lb->czForceLabel,        subset);
  new_dw->allocateAndPut(czTopMat,       d_lb->czTopMatLabel,       subset);
  new_dw->allocateAndPut(czBotMat,       d_lb->czBotMatLabel,       subset);
  new_dw->allocateAndPut(czFailed,       d_lb->czFailedLabel,       subset);
  
  return subset;
}

//__________________________________
//
particleIndex 
CohesiveZone::countCohesiveZones(const Patch* patch, const string filename)
{
  particleIndex sum = 0;

  if(filename!="") {
    std::ifstream is(filename.c_str());
    if (!is ){
      throw ProblemSetupException("ERROR Opening cohesive zone file "+filename+" in countCohesiveZones\n",
                                  __FILE__, __LINE__);
    }

    // Field for position, normal, tangential and length.
    // Everything else is assumed to be zero.
    double f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,mt,mb;
    while(is >> f1 >> f2 >> f3 >> f4 >> f5 >> f6 >> f7 >> f8 >> f9 >> f10 >> mt >> mb){
      //cout << f1 << " " << f2 << " " << f3 << endl;
      if(patch->containsPoint(Point(f1,f2,f3))){
        sum++;
      }
    }
    is.close();
  }

  return sum;
}
//__________________________________
//
vector<const VarLabel* > CohesiveZone::returnCohesiveZoneState()
{
  return d_cz_state;
}
//__________________________________
//
vector<const VarLabel* > CohesiveZone::returnCohesiveZoneStatePreReloc()
{
  return d_cz_state_preReloc;
}
//__________________________________
//
void CohesiveZone::registerPermanentCohesiveZoneState(CZMaterial* czmat)
{
  d_cz_state.push_back(d_lb->czLengthLabel);
  d_cz_state_preReloc.push_back(d_lb->czLengthLabel_preReloc);

  d_cz_state.push_back(d_lb->czNormLabel);
  d_cz_state_preReloc.push_back(d_lb->czNormLabel_preReloc);

  d_cz_state.push_back(d_lb->czTangLabel);
  d_cz_state_preReloc.push_back(d_lb->czTangLabel_preReloc);

  d_cz_state.push_back(d_lb->czDispTopLabel);
  d_cz_state_preReloc.push_back(d_lb->czDispTopLabel_preReloc);

  d_cz_state.push_back(d_lb->czDispBottomLabel);
  d_cz_state_preReloc.push_back(d_lb->czDispBottomLabel_preReloc);

  d_cz_state.push_back(d_lb->czSeparationLabel);
  d_cz_state_preReloc.push_back(d_lb->czSeparationLabel_preReloc);

  d_cz_state.push_back(d_lb->czForceLabel);
  d_cz_state_preReloc.push_back(d_lb->czForceLabel_preReloc);

  d_cz_state.push_back(d_lb->czTopMatLabel);
  d_cz_state_preReloc.push_back(d_lb->czTopMatLabel_preReloc);

  d_cz_state.push_back(d_lb->czBotMatLabel);
  d_cz_state_preReloc.push_back(d_lb->czBotMatLabel_preReloc);

  d_cz_state.push_back(d_lb->czFailedLabel);
  d_cz_state_preReloc.push_back(d_lb->czFailedLabel_preReloc);

  d_cz_state.push_back(d_lb->czIDLabel);
  d_cz_state_preReloc.push_back(d_lb->czIDLabel_preReloc);
}
//__________________________________
//
void CohesiveZone::scheduleInitialize(const LevelP& level, 
                                      SchedulerP& sched,
                                      CZMaterial* czmat)
{
  Task* t = scinew Task("CohesiveZone::initialize",
                  this, &CohesiveZone::initialize);

  MaterialSubset* zeroth_matl = scinew MaterialSubset();
  zeroth_matl->add(0);
  zeroth_matl->addReference();

  t->computes(d_lb->pXLabel);
  t->computes(d_lb->czLengthLabel);
  t->computes(d_lb->czNormLabel);
  t->computes(d_lb->czTangLabel);
  t->computes(d_lb->czDispTopLabel);
  t->computes(d_lb->czDispBottomLabel);
  t->computes(d_lb->czSeparationLabel);
  t->computes(d_lb->czForceLabel);
  t->computes(d_lb->czTopMatLabel);
  t->computes(d_lb->czBotMatLabel);
  t->computes(d_lb->czFailedLabel);
  t->computes(d_lb->czIDLabel);
  t->computes(d_lb->pCellNACZIDLabel,zeroth_matl);

  vector<int> m(1);
  m[0] = czmat->getDWIndex();
  MaterialSet* cz_matl_set = scinew MaterialSet();
  cz_matl_set->addAll(m);
  cz_matl_set->addReference();

  sched->addTask(t, level->eachPatch(), cz_matl_set);

  // The task will have a reference to zeroth_matl
  if (zeroth_matl->removeReference())
    delete zeroth_matl; // shouln't happen, but...
}

//__________________________________
//
void CohesiveZone::initialize(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* cz_matls,
                              DataWarehouse* ,
                              DataWarehouse* new_dw)
{
  particleIndex totalCZs=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

//  printTask(patches, patch,cout_doing,"Doing initialize for CohesiveZones\t");

    CCVariable<short int> cellNACZID;
    new_dw->allocateAndPut(cellNACZID, d_lb->pCellNACZIDLabel, 0, patch);
    cellNACZID.initialize(0);

    for(int m=0;m<cz_matls->size();m++){
      CZMaterial* cz_matl = d_sharedState->getCZMaterial( m );
      string filename = cz_matl->getCohesiveFilename();
      particleIndex numCZs = countCohesiveZones(patch,filename);
      totalCZs+=numCZs;

      cout << "Total CZs " << totalCZs << endl;

      createCohesiveZones(cz_matl, numCZs, cellNACZID, patch, new_dw,filename);
    }
  }

}
