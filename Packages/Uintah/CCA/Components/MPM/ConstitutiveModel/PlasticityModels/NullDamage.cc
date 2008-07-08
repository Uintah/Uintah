
#include "NullDamage.h"
#include <cmath>

using namespace Uintah;
using namespace SCIRun;

NullDamage::NullDamage()
{
} 
         
NullDamage::NullDamage(ProblemSpecP& )
{
} 
         
NullDamage::NullDamage(const NullDamage* )
{
} 
         
NullDamage::~NullDamage()
{
}

void NullDamage::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP damage_ps = ps->appendChild("damage_model");
  damage_ps->setAttribute("type","null");
}

         
inline double 
NullDamage::initialize()
{
  return 0.0;
}

inline bool
NullDamage:: hasFailed(double )
{
  return false;
}
    
double 
NullDamage::computeScalarDamage(const double& ,
                                const Matrix3& ,
                                const double& ,
                                const double& ,
                                const MPMMaterial*,
                                const double& ,
                                const double& )
{
  return 0.0;
}
 
