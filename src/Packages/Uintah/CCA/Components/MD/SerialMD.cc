
#include <Packages/Uintah/CCA/Components/MD/SerialMD.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SoleVariable.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>

#include <iostream>
#include <fstream>

#include <Packages/Uintah/CCA/Components/MD/MDLabel.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


SerialMPM::SerialMPM(const ProcessorGroup* myworld) :
  UintahParallelComponent(myworld)
{
}

SerialMD::~SerialMD()
{
}

void SerialMD::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			     SimulationStateP& sharedState)
{
   d_sharedState = sharedState;

   Problem prob_description;
   prob_description.preProcessor(prob_spec, grid, d_sharedState);

   cerr << "Number of velocity fields = " << d_sharedState->getNumVelFields()
	<< std::endl;
}

void SerialMD::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& dw)
{
}

void SerialMD::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&,
					      DataWarehouseP&)
{
}

void SerialMD::scheduleTimeAdvance(double /*t*/, double /*dt*/,
				    const LevelP&         level,
				          SchedulerP&     sched,
				          DataWarehouseP& old_dw, 
				          DataWarehouseP& new_dw)
{
}

