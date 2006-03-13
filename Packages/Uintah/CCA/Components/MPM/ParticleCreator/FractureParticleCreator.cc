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
