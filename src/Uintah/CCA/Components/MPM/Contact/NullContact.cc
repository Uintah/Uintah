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



// NullContact.cc
// One of the derived Contact classes.  This particular
// class is used when no contact is desired.  This would
// be used for example when a single velocity field is
// present in the problem, so doing contact wouldn't make
// sense.
#include <Packages/Uintah/CCA/Components/MPM/Contact/NullContact.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
using namespace Uintah;

NullContact::NullContact(const ProcessorGroup* myworld,
                         SimulationStateP& d_sS,
			 MPMLabel* Mlb,MPMFlags* MFlags)
  : Contact(myworld, Mlb, MFlags, 0)
{
  // Constructor
  d_sharedState = d_sS;
  lb = Mlb;
  flag = MFlags;

}

NullContact::~NullContact()
{
}

void NullContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type","null");
  d_matls.outputProblemSpec(contact_ps);
}


void NullContact::exMomInterpolated(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* /*old_dw*/,
				    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      NCVariable<Vector> gvelocity;
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      new_dw->getModifiable(   gvelocity, lb->gVelocityLabel, dwi, patch);
    }
  }
}

void NullContact::exMomIntegrated(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* matls,
				    DataWarehouse* /*old_dw*/,
				    DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    for(int m=0;m<matls->size();m++){
      NCVariable<Vector> gv_star;
      NCVariable<Vector> gacc;
      NCVariable<double> frictionalWork;

      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      new_dw->getModifiable(gv_star, lb->gVelocityStarLabel,        dwi, patch);
      new_dw->getModifiable(gacc,    lb->gAccelerationLabel,        dwi, patch);
      new_dw->getModifiable(frictionalWork,lb->frictionalWorkLabel, dwi,
                            patch);
    }
  }
}

void NullContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
						const PatchSet* patches,
						const MaterialSet* ms)
{
  Task * t = scinew Task("NullContact::exMomInterpolated", this, &NullContact::exMomInterpolated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->modifies(lb->gVelocityLabel, mss);
  
  sched->addTask(t, patches, ms);
}

void NullContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
					     const PatchSet* patches,
					     const MaterialSet* ms) 
{
  Task * t = scinew Task("NullContact::exMomIntegrated", this, &NullContact::exMomIntegrated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->modifies(lb->gVelocityStarLabel, mss);
  t->modifies(lb->gAccelerationLabel, mss);
  t->modifies(lb->frictionalWorkLabel, mss);
  
  sched->addTask(t, patches, ms);
}
