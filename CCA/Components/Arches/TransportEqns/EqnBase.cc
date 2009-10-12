#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>

using namespace std;
using namespace Uintah;

EqnBase::EqnBase(ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName):
d_fieldLabels(fieldLabels), d_timeIntegrator(timeIntegrator), d_eqnName(eqnName),
d_doClipping(false), d_doLowClip(false), d_doHighClip(false), d_lowClip(-999999), d_highClip(-999999), d_smallClip(-999999),
d_constant_init(0.0), d_step_dir("x"), d_step_start(0.0), d_step_end(0.0), d_step_cellstart(0.0), d_step_cellend(0.0), d_step_value(0.0),
b_stepUsesCellLocation(false), b_stepUsesPhysicalLocation(false)
{
  d_boundaryCond = scinew BoundaryCondition_new( d_fieldLabels ); 
  d_disc = scinew Discretization_new(); 
}

EqnBase::~EqnBase()
{
  delete(d_boundaryCond);
  delete(d_disc);
}
