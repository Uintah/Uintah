
#include <Packages/Uintah/CCA/Ports/MPMCFDInterface.h>

#include <iostream>

using namespace Uintah;

MPMCFDInterface::MPMCFDInterface()
{
  d_analyze = NULL;
}

MPMCFDInterface::~MPMCFDInterface()
{
}

void MPMCFDInterface::setAnalyze(PatchDataAnalyze* analyze)
{
  std::cout<<"MPMCFDInterface::setAnalyze"<<std::endl;
  d_analyze = analyze;
}
