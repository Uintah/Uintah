#include "NoneCheck.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>


using namespace Uintah;
using namespace std;

NoneCheck::NoneCheck(ProblemSpecP& )
{
}

NoneCheck::NoneCheck(const NoneCheck*)
{
}

NoneCheck::~NoneCheck()
{
}

void NoneCheck::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP stability_ps = ps->appendChild("stability_check");
  stability_ps->setAttribute("type","none");

}
	 
bool 
NoneCheck::checkStability(const Matrix3& ,
                          const Matrix3& deformRate ,
                          const TangentModulusTensor& Cep ,
                          Vector& )
{
  return true;
}

