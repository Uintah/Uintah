#include "DruckerCheck.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>


using namespace Uintah;
using namespace std;

DruckerCheck::DruckerCheck(ProblemSpecP& ps)
{
}

DruckerCheck::~DruckerCheck()
{
}
	 
bool 
DruckerCheck::checkStability(const Matrix3& ,
			     const Matrix3& deformRate ,
			     const TangentModulusTensor& Cep ,
			     Vector& )
{
  // Calculate the stress rate
  Matrix3 stressRate(0.0);
  Cep.contract(deformRate, stressRate);

  cout << "Deform Rate = \n" << deformRate << endl;
  cout << "Cep = \n" << Cep ;
  cout << "Stress Rate = \n" << stressRate << endl;

  double val = stressRate.Contract(deformRate);
  cout << "val = " << val << endl << endl;
  if (val > 0.0) return false;
  return true;
}

