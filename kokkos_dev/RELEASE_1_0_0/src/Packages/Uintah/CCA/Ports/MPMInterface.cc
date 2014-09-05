
#include <Packages/Uintah/CCA/Ports/MPMInterface.h>

using namespace Uintah;

MPMInterface::MPMInterface()
{
  d_analyze = NULL;
}

MPMInterface::~MPMInterface()
{
}

void MPMInterface::setAnalyze(PatchDataAnalyze* analyze)
{
  d_analyze = analyze;
}

