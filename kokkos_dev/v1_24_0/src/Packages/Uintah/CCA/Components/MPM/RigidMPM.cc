#include <Packages/Uintah/CCA/Components/MPM/RigidMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Packages/Uintah/CCA/Components/MPM/Contact/ContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContact.h>
#include <Packages/Uintah/CCA/Components/MPM/ThermalContact/ThermalContactFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/PressureBC.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/DebugStream.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;

using namespace std;

#define MAX_BASIS 27
#undef INTEGRAL_TRACTION

static DebugStream cout_doing("RIGID_MPM", false);

// From ThreadPool.cc:  Used for syncing cerr'ing so it is easier to read.
extern Mutex cerrLock;

RigidMPM::RigidMPM(const ProcessorGroup* myworld) :
  SerialMPM(myworld)
{
}

RigidMPM::~RigidMPM()
{
}

void RigidMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
                       SimulationStateP& sharedState)
{

  SerialMPM::problemSetup(prob_spec, grid, sharedState);
  ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
  if(cfd_ps){
    cout << "\n__________________________________"<< endl;
    cout << "  W A R N I N G :  " << endl;
    cout << "  You must use stiff MPM material properties" << endl;
    cout << "  to get the correct pressure solution in rmpmice" << endl;
    cout << "__________________________________\n"<< endl;
  }
}

void RigidMPM::computeStressTensor(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse* old_dw,
				    DataWarehouse* new_dw)
{

  cout_doing <<"Doing computeStressTensor " <<"\t\t\t\t RigidMPM"<< endl;

  for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
    MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial(m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    cm->carryForward(patches, mpm_matl, old_dw, new_dw);
  }

  new_dw->put(delt_vartype(getLevel(patches)->adjustDelt(999.0)), 
              lb->delTLabel);

}

void RigidMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::computeInternalForce",
                    this, &RigidMPM::computeInternalForce);
  sched->addTask(t, patches, matls);
}

void RigidMPM::computeInternalForce(const ProcessorGroup*,
				    const PatchSubset* patches,
				    const MaterialSubset* ,
				    DataWarehouse*,
				    DataWarehouse*)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing computeInternalForce on patch " << patch->getID()
	       <<"\t\t\t RigidMPM"<< endl;

  }
}

void RigidMPM::scheduleSolveEquationsMotion(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::solveEquationsMotion",
                    this, &RigidMPM::solveEquationsMotion);

  t->computes(lb->gAccelerationLabel);
  sched->addTask(t, patches, matls);
}

void RigidMPM::solveEquationsMotion(const ProcessorGroup*,
				     const PatchSubset* patches,
				     const MaterialSubset*,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    cout_doing <<"Doing solveEquationsMotion on patch " << patch->getID()
	       <<"\t\t\t RigidMPM"<< endl;

    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
 
      // Create variables for the results
      NCVariable<Vector> acceleration;
      new_dw->allocateAndPut(acceleration, lb->gAccelerationLabel, dwi, patch);
      acceleration.initialize(Vector(0.,0.,0.));

    }
  }
}

