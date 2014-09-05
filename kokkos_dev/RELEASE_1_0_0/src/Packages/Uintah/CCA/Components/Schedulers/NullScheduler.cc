

#include <Packages/Uintah/CCA/Components/Schedulers/NullScheduler.h>
#include <Packages/Uintah/CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <Packages/Uintah/CCA/Ports/LoadBalancer.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/Core/Grid/ReductionVariable.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static DebugStream dbg("NullScheduler", false);

NullScheduler::NullScheduler(const ProcessorGroup* myworld,
			     Output* oport)
   : SchedulerCommon(myworld, oport)
{
   d_generation = 0;
   delt = scinew VarLabel("delT",
    ReductionVariable<double, Reductions::Min<double> >::getTypeDescription());
   firstTime=true;
}

NullScheduler::~NullScheduler()
{
}

void 
NullScheduler::advanceDataWarehouse(const GridP& grid)
{
  if(!dw[1])
    dw[1]=scinew OnDemandDataWarehouse(d_myworld, 0, grid);
}

void
NullScheduler::execute(const ProcessorGroup * pg)
{
  ASSERT(dt != 0);
  if(firstTime){
    firstTime=false;
    dw[Task::NewDW]->put(delt_vartype(1.0), delt);
  }
}

void
NullScheduler::scheduleParticleRelocation(const LevelP&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const VarLabel*,
					  const vector<vector<const VarLabel*> >&,
					  const MaterialSet*)
{
}
