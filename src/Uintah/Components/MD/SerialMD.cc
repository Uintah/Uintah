/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/MPM/SerialMPM.h>
#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/ReductionVariable.h>
#include <Uintah/Grid/SimulationState.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Exceptions/ParameterNotFound.h>

#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>

#include <iostream>
#include <fstream>

#include <Uintah/Components/MPM/MPMLabel.h>

using namespace Uintah;
using namespace Uintah::MPM;

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
using SCICore::Geometry::Dot;
using SCICore::Math::Min;
using SCICore::Math::Max;
using namespace std;


SerialMPM::SerialMPM( int MpiRank, int MpiProcesses ) :
  UintahParallelComponent( MpiRank, MpiProcesses )
{
}

SerialMPM::~SerialMPM()
{
}

void SerialMPM::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
			     SimulationStateP& sharedState)
{
}

void SerialMPM::scheduleInitialize(const LevelP& level,
				   SchedulerP& sched,
				   DataWarehouseP& dw)
{
}

void SerialMPM::scheduleComputeStableTimestep(const LevelP&,
					      SchedulerP&,
					      DataWarehouseP&)
{
}

void SerialMPM::scheduleTimeAdvance(double /*t*/, double /*dt*/,
				    const LevelP&         level,
				          SchedulerP&     sched,
				          DataWarehouseP& old_dw, 
				          DataWarehouseP& new_dw)
{
}

// $Log$
// Revision 1.1  2000/06/09 18:02:08  tan
// Create SerialMD to do molecular dynamics simulations.
//
