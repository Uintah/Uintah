#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>

using namespace Uintah;

RegridderCommon::RegridderCommon(ProcessorGroup* pg) : Regridder(), UintahParallelComponent(pg)
{

}

RegridderCommon::~RegridderCommon()
{

}

bool RegridderCommon::needRecompile(double time, double delt, const GridP& grid)
{
  bool retval = newGrid;
  newGrid = false;
  return retval;
}

void RegridderCommon::problemSetup(const ProblemSpecP& params)
{

}
