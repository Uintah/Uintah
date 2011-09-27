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


#include <CCA/Components/Angio/AngioParticleCreator.h>
#include <CCA/Components/Angio/AngioMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/SmoothGeomPiece.h>
#include <Core/Labels/AngioLabel.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <fstream>
#include <iostream>

using namespace Uintah;
using std::vector;
using std::cerr;
using std::ofstream;

AngioParticleCreator::AngioParticleCreator(AngioMaterial* matl,
                                           AngioFlags* flags)
{
  d_lb = scinew AngioLabel();

  registerPermanentParticleState();
}

AngioParticleCreator::~AngioParticleCreator()
{
  delete d_lb;
}

particleIndex 
AngioParticleCreator::countParticles(const Patch* patch,
                                     std::string init_frag_file)
{
  particleIndex sum = 0;

  std::ifstream iff(init_frag_file.c_str());
  if (!iff ){
    throw ProblemSetupException("ERROR: opening init frag file '"+init_frag_file+"'\n", __FILE__, __LINE__);
  }
  int num_frags=0;
  double px0,py0,pz0,px1,py1,pz1;
  while(iff >> px0  >> py0 >> pz0 >> px1 >> py1 >> pz1){
    Point p(px0,py0,pz0);
    if(patch->containsPoint(p)){
      num_frags++;
    }
  }
  iff.close();

  sum+=num_frags;

  return sum;
}

ParticleSubset* 
AngioParticleCreator::createParticles(AngioMaterial* matl,
                                      std::string init_frag_file,
                                      particleIndex numParticles,
                                      CCVariable<short int>& cellNAPID,
                                      const Patch* patch,DataWarehouse* new_dw)
{
  int dwi = matl->getDWIndex();
  ParticleSubset* subset = allocateVariables(numParticles,dwi,patch,new_dw);

  particleIndex start = 0;

  std::ifstream iff(init_frag_file.c_str());
  if (!iff ){
    throw ProblemSetupException("ERROR: opening init frag file '"+init_frag_file+"'\n", __FILE__, __LINE__);
  }

  double px1,py1,pz1,px0,py0,pz0;
  particleIndex count = 0;
  IntVector cell_idx;

  while(iff >> px0  >> py0 >> pz0 >> px1 >> py1 >> pz1){
    Point p0(px0,py0,pz0);
    Point p1(px1,py1,pz1);
    if(patch->findCell(p0,cell_idx)){
      particleIndex pidx = start+count;      

      initializeParticle(patch,matl,p0,p1,cell_idx,pidx,cellNAPID);

      count++;
    }
  }
  start += count;

  return subset;
}

ParticleSubset* 
AngioParticleCreator::allocateVariables(particleIndex numParticles, 
                                   int dwi, const Patch* patch,
                                   DataWarehouse* new_dw)
{
  ParticleSubset* subset = new_dw->createParticleSubset(numParticles,dwi,patch);

  new_dw->allocateAndPut(position0,      d_lb->pXLabel,             subset);
  new_dw->allocateAndPut(growth,         d_lb->pGrowthLabel,        subset);
  new_dw->allocateAndPut(length,         d_lb->pLengthLabel,        subset);
  new_dw->allocateAndPut(phi,            d_lb->pPhiLabel,           subset);
  new_dw->allocateAndPut(radius,         d_lb->pRadiusLabel,        subset);
  new_dw->allocateAndPut(tofb,           d_lb->pTimeOfBirthLabel,   subset);
  new_dw->allocateAndPut(recentbranch,   d_lb->pRecentBranchLabel,  subset);
  new_dw->allocateAndPut(pmass,          d_lb->pMassLabel,          subset);
  new_dw->allocateAndPut(pvolume,        d_lb->pVolumeLabel,        subset);
  new_dw->allocateAndPut(tip0,           d_lb->pTip0Label,          subset);
  new_dw->allocateAndPut(tip1,           d_lb->pTip1Label,          subset);
  new_dw->allocateAndPut(parent,         d_lb->pParentLabel,        subset);
  new_dw->allocateAndPut(pparticleID,    d_lb->pParticleIDLabel,    subset);

  return subset;
}

void AngioParticleCreator::allocateVariablesAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*,
                                   ParticleVariableBase*>* newState,
                               vector<Point>& x_new, vector<Vector>& growth_new,
                               vector<double>& l_new, vector<double>& rad_new,
                               vector<double>& ang_new,vector<double>& t_new,
                               vector<double>& r_b_new,
                               vector<double>& pmass_new,
                               vector<double>& pvol_new, vector<int>& pt0_new,
                               vector<int>& pt1_new,     vector<int>& par_new,
                               vector<IntVector>& vcell_idx,
                               CCVariable<short int>& cellNAPID)
{
  ParticleSubset::iterator n;

  new_dw->allocateTemporary(position0,      addset);
  new_dw->allocateTemporary(growth,         addset);
  new_dw->allocateTemporary(length,         addset);
  new_dw->allocateTemporary(phi,            addset);
  new_dw->allocateTemporary(radius,         addset);
  new_dw->allocateTemporary(tofb,           addset);
  new_dw->allocateTemporary(pmass,          addset);
  new_dw->allocateTemporary(pvolume,        addset);
  new_dw->allocateTemporary(recentbranch,   addset);
  new_dw->allocateTemporary(tip0,           addset);
  new_dw->allocateTemporary(tip1,           addset);
  new_dw->allocateTemporary(parent,         addset);
  new_dw->allocateTemporary(pparticleID,    addset);

  n = addset->begin();
  int ic=0;
  for (n = addset->begin(); n != addset->end(); n++, ic++) {
    position0[*n]      = x_new[ic];
    growth[*n]         = growth_new[ic];
    length[*n]         = l_new[ic];
    phi[*n]            = ang_new[ic];
    radius[*n]         = rad_new[ic];
    tofb[*n]           = t_new[ic];
    recentbranch[*n]   = r_b_new[ic];
    tip0[*n]           = pt0_new[ic];
    tip1[*n]           = pt1_new[ic];
    pmass[*n]          = pmass_new[ic];
    pvolume[*n]        = pvol_new[ic];
    parent[*n]         = par_new[ic];

    // Figure out the new particle's ID
    ASSERT(vcell_idx[ic].x() <= 0xffff && 
           vcell_idx[ic].y() <= 0xffff && 
           vcell_idx[ic].z() <= 0xffff);
         
    long64 cellID = ((long64)vcell_idx[ic].x() << 16) | 
                    ((long64)vcell_idx[ic].y() << 32) | 
                    ((long64)vcell_idx[ic].z() << 48);
                  
    short int& myCellNAPID = cellNAPID[vcell_idx[ic]];
    pparticleID[*n] = (cellID | (long64) myCellNAPID);
    ASSERT(myCellNAPID < 0x7fff);
    myCellNAPID++;
  }
  
  (*newState)[d_lb->pXLabel]               =position0.clone();
  (*newState)[d_lb->pGrowthLabel]          =growth.clone();
  (*newState)[d_lb->pLengthLabel]          =length.clone();
  (*newState)[d_lb->pPhiLabel]             =phi.clone();
  (*newState)[d_lb->pRadiusLabel]          =radius.clone();
  (*newState)[d_lb->pTimeOfBirthLabel]     =tofb.clone();
  (*newState)[d_lb->pRecentBranchLabel]    =recentbranch.clone();
  (*newState)[d_lb->pTip0Label]            =tip0.clone();
  (*newState)[d_lb->pTip1Label]            =tip1.clone();
  (*newState)[d_lb->pMassLabel]            =pmass.clone();
  (*newState)[d_lb->pVolumeLabel]          =pvolume.clone();
  (*newState)[d_lb->pParentLabel]          =parent.clone();
  (*newState)[d_lb->pParticleIDLabel]      =pparticleID.clone();
}

void AngioParticleCreator::initializeParticle(const Patch* patch,
                                              AngioMaterial* matl,
                                              Point p0, Point p1,
                                              IntVector cell_idx,
                                              particleIndex i,
                                              CCVariable<short int>& cellNAPID)
{
  const double pi = 3.14159265359;
  position0[i]    = p0;
  growth[i]       = p1-p0;
  radius[i]       = 1.0;
  phi[i]          = atan((p1.y() - p0.y())/(p1.x() - p0.x()));
  length[i]       = growth[i].length();
  pvolume[i]      = length[i]*pi*radius[i]*radius[i];
  tofb[i]         = 0;
  recentbranch[i] = 0;
  tip0[i]         = -1;
  tip1[i]         = 1;
  pmass[i]        = matl->getInitialDensity()*pvolume[i];
  parent[i]       = i;

  ASSERT(cell_idx.x() <= 0xffff && 
         cell_idx.y() <= 0xffff && 
         cell_idx.z() <= 0xffff);
         
  long64 cellID = ((long64)cell_idx.x() << 16) | 
                  ((long64)cell_idx.y() << 32) | 
                  ((long64)cell_idx.z() << 48);
                  
  short int& myCellNAPID = cellNAPID[cell_idx];
  pparticleID[i] = (cellID | (long64) myCellNAPID);
  ASSERT(myCellNAPID < 0x7fff);
  myCellNAPID++;
}

vector<const VarLabel* > AngioParticleCreator::returnParticleState()
{
  return particle_state;
}

vector<const VarLabel* > AngioParticleCreator::returnParticleStatePreReloc()
{
  return particle_state_preReloc;
}

void AngioParticleCreator::registerPermanentParticleState()
{
  particle_state.push_back(d_lb->pGrowthLabel);
  particle_state_preReloc.push_back(d_lb->pGrowthLabel_preReloc);

  particle_state.push_back(d_lb->pLengthLabel);
  particle_state_preReloc.push_back(d_lb->pLengthLabel_preReloc);

  particle_state.push_back(d_lb->pPhiLabel);
  particle_state_preReloc.push_back(d_lb->pPhiLabel_preReloc);

  particle_state.push_back(d_lb->pRadiusLabel);
  particle_state_preReloc.push_back(d_lb->pRadiusLabel_preReloc);

  particle_state.push_back(d_lb->pTimeOfBirthLabel);
  particle_state_preReloc.push_back(d_lb->pTimeOfBirthLabel_preReloc);

  particle_state.push_back(d_lb->pRecentBranchLabel);
  particle_state_preReloc.push_back(d_lb->pRecentBranchLabel_preReloc);

  particle_state.push_back(d_lb->pTip0Label);
  particle_state_preReloc.push_back(d_lb->pTip0Label_preReloc);

  particle_state.push_back(d_lb->pTip1Label);
  particle_state_preReloc.push_back(d_lb->pTip1Label_preReloc);

  particle_state.push_back(d_lb->pMassLabel);
  particle_state_preReloc.push_back(d_lb->pMassLabel_preReloc);

  particle_state.push_back(d_lb->pVolumeLabel);
  particle_state_preReloc.push_back(d_lb->pVolumeLabel_preReloc);

  particle_state.push_back(d_lb->pParentLabel);
  particle_state_preReloc.push_back(d_lb->pParentLabel_preReloc);

  particle_state.push_back(d_lb->pParticleIDLabel);
  particle_state_preReloc.push_back(d_lb->pParticleIDLabel_preReloc);
}
