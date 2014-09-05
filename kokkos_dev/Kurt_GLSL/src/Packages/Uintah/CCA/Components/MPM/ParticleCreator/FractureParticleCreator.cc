#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/FractureParticleCreator.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>

using namespace Uintah;
using std::vector;

FractureParticleCreator::FractureParticleCreator(MPMMaterial* matl,
                                                 MPMFlags* flags)

  :  ParticleCreator(matl,flags)
{
  registerPermanentParticleState(matl);
}

FractureParticleCreator::~FractureParticleCreator()
{
}

ParticleSubset* FractureParticleCreator::createParticles(MPMMaterial* matl,
                                              particleIndex numParticles,
                                              CCVariable<short int>& cellNAPID,
                                              const Patch* patch,
                                              DataWarehouse* new_dw,
                                              vector<GeometryObject*>& d_geom_objs)
{

 ParticleSubset* subset = ParticleCreator::createParticles(matl,numParticles,
                                                           cellNAPID,patch,
                                                           new_dw,
                                                           d_geom_objs);

 //ParticleVariable<Point> position0;
 //constParticleVariable<Point> position;

 //new_dw->allocateAndPut(position0,lb->pX0Label,subset);
 //new_dw->get(position,lb->pXLabel,subset);

 //position0.copyData(position);

 return subset;
}

particleIndex 
FractureParticleCreator::countParticles(const Patch* patch,
                                        vector<GeometryObject*>& d_geom_objs) 
{

  return ParticleCreator::countParticles(patch,d_geom_objs);
}

particleIndex 
FractureParticleCreator::countAndCreateParticles(const Patch* patch,
                                                 GeometryObject* obj) 
{

  return ParticleCreator::countAndCreateParticles(patch,obj);
}


void
FractureParticleCreator::registerPermanentParticleState(MPMMaterial* /*matl*/)

{
  //particle_state.push_back(lb->pX0Label);
  //particle_state_preReloc.push_back(lb->pX0Label_preReloc);

  vector<const VarLabel*>::iterator r1,r2;
  r1 = find(particle_state.begin(), particle_state.end(),d_lb->pErosionLabel);
  particle_state.erase(r1);

  r2 = find(particle_state_preReloc.begin(), particle_state_preReloc.end(),
         d_lb->pErosionLabel_preReloc);
  particle_state_preReloc.erase(r2);
}


void 
FractureParticleCreator::applyForceBC(const Vector& dxpp, 
                                      const Point& pp,
                                      const double& pMass, 
                                      Vector& pExtForce)
{
  for (int i = 0; i<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); i++){
    string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[i]->getType();
        
    //cerr << " BC Type = " << bcs_type << endl;
    if (bcs_type == "Force") {
      ForceBC* bc = dynamic_cast<ForceBC*>
        (MPMPhysicalBCFactory::mpmPhysicalBCs[i]);

      Box bcBox;
        bcBox = Box(bc->getLowerRange(), bc->getUpperRange());

      //cerr << "BC Box = " << bcBox << " Point = " << pp << endl;
      if(bcBox.contains(pp)) {
        pExtForce = bc->getForceDensity() * pMass;
        //cerr << "External Force on Particle = " << pExtForce 
        //     << " Force Density = " << bc->getForceDensity() 
        //     << " Particle Mass = " << pMass << endl;
      }
    } 
  }
}
