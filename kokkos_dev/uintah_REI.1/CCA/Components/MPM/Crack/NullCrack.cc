#include <Packages/Uintah/CCA/Components/MPM/Crack/NullCrack.h>


using namespace Uintah;


NullCrack::NullCrack(ProblemSpecP& ps)
{
}


NullCrack::~NullCrack()
{
}


void NullCrack::readCrack(ProblemSpecP& arc_ps)
{
}


void NullCrack::outputInitialCrackPlane(int i)
{
}

void NullCrack::discretize(int& nstart0,vector<Point>& cx,
                           vector<IntVector>& ce,vector<int>& SegNodes)
{
}

