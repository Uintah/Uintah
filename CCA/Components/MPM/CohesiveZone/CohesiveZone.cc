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


#include <CCA/Components/MPM/CohesiveZone/CohesiveZone.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <fstream>
#include <iostream>

using namespace Uintah;
using std::vector;
using std::cerr;
using std::ofstream;

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

ParticleSubset* 
CohesiveZone::createCohesiveZones(CZMaterial* matl,
                                 particleIndex numCohesiveZones,
                                 CCVariable<short int>& cellNAPID,
                                 const Patch* patch,DataWarehouse* new_dw,
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

    // Field for position, normal, tangential and length.
    // Everything else is assumed to be zero.
    double p1,p2,p3,l4,n5,n6,n7,t8,t9,t10;
    int mt, mb;
    while(is >> p1 >> p2 >> p3 >> l4 >> n5 >> n6 >> n7 >> t8 >> t9 >> t10 >> mb >> mt){
      Point pos = Point(p1,p2,p3);
        IntVector cell_idx;
      if(patch->findCell(pos,cell_idx)){
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

        // Figure out unique ID for the CZ
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
    }
    is.close();
  }

  return subset;
}

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
  
  return subset;
}

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

vector<const VarLabel* > CohesiveZone::returnCohesiveZoneState()
{
  return cz_state;
}

vector<const VarLabel* > CohesiveZone::returnCohesiveZoneStatePreReloc()
{
  return cz_state_preReloc;
}

void CohesiveZone::registerPermanentCohesiveZoneState(CZMaterial* czmat)
{
  cz_state.push_back(d_lb->czLengthLabel);
  cz_state_preReloc.push_back(d_lb->czLengthLabel_preReloc);

  cz_state.push_back(d_lb->czNormLabel);
  cz_state_preReloc.push_back(d_lb->czNormLabel_preReloc);

  cz_state.push_back(d_lb->czTangLabel);
  cz_state_preReloc.push_back(d_lb->czTangLabel_preReloc);

  cz_state.push_back(d_lb->czDispTopLabel);
  cz_state_preReloc.push_back(d_lb->czDispTopLabel_preReloc);

  cz_state.push_back(d_lb->czDispBottomLabel);
  cz_state_preReloc.push_back(d_lb->czDispBottomLabel_preReloc);

  cz_state.push_back(d_lb->czSeparationLabel);
  cz_state_preReloc.push_back(d_lb->czSeparationLabel_preReloc);

  cz_state.push_back(d_lb->czForceLabel);
  cz_state_preReloc.push_back(d_lb->czForceLabel_preReloc);

  cz_state.push_back(d_lb->czTopMatLabel);
  cz_state_preReloc.push_back(d_lb->czTopMatLabel_preReloc);

  cz_state.push_back(d_lb->czBotMatLabel);
  cz_state_preReloc.push_back(d_lb->czBotMatLabel_preReloc);

  cz_state.push_back(d_lb->czIDLabel);
  cz_state_preReloc.push_back(d_lb->czIDLabel_preReloc);
}

void CohesiveZone::scheduleInitialize(const LevelP& level, SchedulerP& sched,
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

void CohesiveZone::initialize(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* ,
                              DataWarehouse* new_dw)
{
  particleIndex totalCZs=0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

//  printTask(patches, patch,cout_doing,"Doing initialize for CohesiveZones\t");

    CCVariable<short int> cellNACZID;
    new_dw->allocateAndPut(cellNACZID, d_lb->pCellNACZIDLabel, 0, patch);
    cellNACZID.initialize(0.);

    for(int m=0;m<matls->size();m++){
      CZMaterial* cz_matl = d_sharedState->getCZMaterial( m );
      string filename = cz_matl->getCohesiveFilename();
      particleIndex numCZs = countCohesiveZones(patch,filename);
      totalCZs+=numCZs;

      cout << "Total CZs " << totalCZs << endl;

      createCohesiveZones(cz_matl, numCZs, cellNACZID, patch, new_dw,filename);
    }
  }

}
