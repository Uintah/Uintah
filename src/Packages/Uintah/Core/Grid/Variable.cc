
#include <Packages/Uintah/Core/Grid/Variable.h>

using namespace Uintah;

Variable::Variable()
{
   d_foreign = false;
}

Variable::~Variable()
{
}

void Variable::setForeign()
{
   d_foreign = true;
}


