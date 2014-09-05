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
