#include <Packages/Uintah/CCA/Components/Regridder/HierarchicalRegridder.h>
#include <Packages/Uintah/Core/Grid/Grid.h>

using namespace Uintah;

HierarchicalRegridder::HierarchicalRegridder(const ProcessorGroup* pg) : RegridderCommon(pg)
{

}

HierarchicalRegridder::~HierarchicalRegridder()
{

}

Grid* HierarchicalRegridder::regrid(Grid* oldgrid, SchedulerP sched)
{

  return oldgrid;
}
