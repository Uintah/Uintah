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
#include <CCA/Components/MPM/MPMFlags.h>
#include <Core/GeometryPiece/GeometryObject.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <CCA/Ports/DataWarehouse.h>
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
#include <fstream>
#include <iostream>

using namespace Uintah;
using std::vector;
using std::cerr;
using std::ofstream;

CohesiveZone::CohesiveZone(CZMaterial* czmat, MPMFlags* flags)
{
  d_lb = scinew MPMLabel();

  d_flags = flags;

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
                                 const Patch* patch,DataWarehouse* new_dw)
{
  // Print the physical boundary conditions
  //  printPhysicalBCs();

  int dwi = matl->getDWIndex();
  ParticleSubset* subset = allocateVariables(numCohesiveZones,dwi,patch,new_dw);

  particleIndex start = 0;

  return subset;
}

ParticleSubset* 
CohesiveZone::allocateVariables(particleIndex numParticles, 
                                   int dwi, const Patch* patch,
                                   DataWarehouse* new_dw)
{

  ParticleSubset* subset = new_dw->createParticleSubset(numParticles,dwi,patch);

#if 0
  new_dw->allocateAndPut(position,       d_lb->pXLabel,             subset);
  new_dw->allocateAndPut(pvelocity,      d_lb->pVelocityLabel,      subset); 
  new_dw->allocateAndPut(pexternalforce, d_lb->pExternalForceLabel, subset);
  new_dw->allocateAndPut(pmass,          d_lb->pMassLabel,          subset);
  new_dw->allocateAndPut(pvolume,        d_lb->pVolumeLabel,        subset);
  new_dw->allocateAndPut(ptemperature,   d_lb->pTemperatureLabel,   subset);
  new_dw->allocateAndPut(pparticleID,    d_lb->pParticleIDLabel,    subset);
  new_dw->allocateAndPut(psize,          d_lb->pSizeLabel,          subset);
  new_dw->allocateAndPut(pfiberdir,      d_lb->pFiberDirLabel,      subset); 
  new_dw->allocateAndPut(perosion,       d_lb->pErosionLabel,       subset); 
  new_dw->allocateAndPut(pdisp,          d_lb->pDispLabel,          subset);
#endif
  
  return subset;
}

void CohesiveZone::createPoints(const Patch* patch, GeometryObject* obj)
{
  GeometryPieceP piece = obj->getPiece();
  Box b2 = patch->getExtraBox();
  IntVector ppc = obj->getNumParticlesPerCell();
  Vector dxpp = patch->dCell()/ppc;
  Vector dcorner = dxpp*0.5;

  for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
    Point lower = patch->nodePosition(*iter) + dcorner;
    IntVector c = *iter;
    
    for(int ix=0;ix < ppc.x(); ix++){
      for(int iy=0;iy < ppc.y(); iy++){
        for(int iz=0;iz < ppc.z(); iz++){
        
          IntVector idx(ix, iy, iz);
          Point p = lower + dxpp*idx;
          if (!b2.contains(p)){
            throw InternalError("Particle created outside of patch?", __FILE__, __LINE__);
          }
        }  // z
      }  // y
    }  // x
  }  // iterator

}


void 
CohesiveZone::initializeCohesiveZone(const Patch* patch,
                                    vector<GeometryObject*>::const_iterator obj,
                                    MPMMaterial* matl,
                                    Point p,
                                    IntVector cell_idx,
                                    particleIndex i,
                                    CCVariable<short int>& cellNACZID)
{
  IntVector ppc = (*obj)->getNumParticlesPerCell();
  Vector dxpp = patch->dCell()/(*obj)->getNumParticlesPerCell();
  czposition[i] = p;

#if 0
  psize[i]    = size;

  pvelocity[i]    = (*obj)->getInitialVelocity();
  ptemperature[i] = (*obj)->getInitialData("temperature");
  pmass[i]        = matl->getInitialDensity()*pvolume[i];
  pdisp[i]        = Vector(0.,0.,0.);
  
  if(d_with_color){
    pcolor[i] = (*obj)->getInitialData("color");
  }
  if(d_artificial_viscosity){
    p_q[i] = 0.;
  }
  
  ptempPrevious[i]  = ptemperature[i];

  Vector pExtForce(0,0,0);
  
  pexternalforce[i] = pExtForce;
  pfiberdir[i]      = matl->getConstitutiveModel()->getInitialFiberDir();
  perosion[i]       = 1.0;
#endif

  ASSERT(cell_idx.x() <= 0xffff && 
         cell_idx.y() <= 0xffff && 
         cell_idx.z() <= 0xffff);
         
  long64 cellID = ((long64)cell_idx.x() << 16) | 
                  ((long64)cell_idx.y() << 32) | 
                  ((long64)cell_idx.z() << 48);
                  
  short int& myCellNACZID = cellNACZID[cell_idx];
  czID[i] = (cellID | (long64) myCellNACZID);
  ASSERT(myCellNACZID < 0x7fff);
  myCellNACZID++;
}

particleIndex 
CohesiveZone::countCohesiveZones(const Patch* patch)
{
  particleIndex sum = 0;

  return sum;
}


particleIndex 
CohesiveZone::countAndCreateCohesiveZones(const Patch* patch, 
                                         GeometryObject* obj)
{
  GeometryPieceP piece = obj->getPiece();
  Box b1 = piece->getBoundingBox();
  Box b2 = patch->getExtraBox();
  Box b = b1.intersect(b2);
  if(b.degenerate()) return 0;
  
  createPoints(patch,obj);
  
  return (particleIndex) 987654321;
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
}
