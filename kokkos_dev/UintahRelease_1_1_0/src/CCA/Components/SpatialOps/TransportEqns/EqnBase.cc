#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/BoundaryCond.h>
#include <CCA/Components/SpatialOps/ExplicitTimeInt.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>

using namespace std;
using namespace Uintah;

EqnBase::EqnBase(Fields* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName) :
  d_fieldLabels( fieldLabels ), d_eqnName( eqnName ), d_timeIntegrator( timeIntegrator )
{
}

EqnBase::~EqnBase()
{
}
 
