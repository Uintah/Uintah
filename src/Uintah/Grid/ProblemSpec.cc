
#include "ProblemSpec.h"
#include <iostream>
using std::cerr;

ProblemSpec::ProblemSpec()
{
}

ProblemSpec::~ProblemSpec()
{
}

double ProblemSpec::getStartTime() const
{
    return 0;
}

double ProblemSpec::getMaximumTime() const
{
    //    return 0.00002;
    //return 1.e-5;
    return .004;
}

const TypeDescription* ProblemSpec::getTypeDescription()
{
    //cerr << "ProblemSpec::getTypeDescription() not done\n";
    return 0;
}
