#include <CCA/Components/MPM/ImpMPMFlags.h>
#include <SCIRun/Core/Util/DebugStream.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream dbg("ImpMPMFlags", false);

ImpMPMFlags::ImpMPMFlags() : MPMFlags()
{

  d_conv_crit_disp   = 1.e-10;
  d_conv_crit_energy = 4.e-10;
  d_forceIncrementFactor = 1.0;
  d_integrator = Implicit;
  d_dynamic = true;

  d_doMechanics = true;
  d_projectHeatSource = false;

  d_temp_solve = false;
  d_interpolateParticleTempToGridEveryStep = true;
}

ImpMPMFlags::~ImpMPMFlags()
{

}

void
ImpMPMFlags::readMPMFlags(ProblemSpecP& ps)
{
  MPMFlags::readMPMFlags(ps);

  ProblemSpecP root = ps->getRootNode();
  ProblemSpecP mpm_flag_ps = root->findBlock("MPM");

  if (!mpm_flag_ps)
    return;


  mpm_flag_ps->get("ProjectHeatSource", d_projectHeatSource);
  mpm_flag_ps->get("DoMechanics", d_doMechanics);
  mpm_flag_ps->get("convergence_criteria_disp",  d_conv_crit_disp);
  mpm_flag_ps->get("convergence_criteria_energy",d_conv_crit_energy);
  mpm_flag_ps->get("dynamic",d_dynamic);
  mpm_flag_ps->getWithDefault("iters_before_timestep_restart",
                              d_max_num_iterations, 25);
  mpm_flag_ps->getWithDefault("num_iters_to_decrease_delT",
                              d_num_iters_to_decrease_delT, 12);
  mpm_flag_ps->getWithDefault("num_iters_to_increase_delT",
                              d_num_iters_to_increase_delT, 4);
  mpm_flag_ps->getWithDefault("delT_decrease_factor",
                              d_delT_decrease_factor, .6);
  mpm_flag_ps->getWithDefault("delT_increase_factor",
                              d_delT_increase_factor, 2);
  
  mpm_flag_ps->get("solver",d_solver_type);
  mpm_flag_ps->get("temperature_solve",d_temp_solve);
  mpm_flag_ps->get("interpolateParticleTempToGridEveryStep",
                  d_interpolateParticleTempToGridEveryStep);

}

void
ImpMPMFlags::outputProblemSpec(ProblemSpecP& ps)
{

  MPMFlags::outputProblemSpec(ps);

  ps->appendElement("ProjectHeatSource", d_projectHeatSource);
  ps->appendElement("DoMechanics", d_doMechanics);
  ps->appendElement("convergence_criteria_disp",  d_conv_crit_disp);
  ps->appendElement("convergence_criteria_energy",d_conv_crit_energy);
  ps->appendElement("dynamic",d_dynamic);
  ps->appendElement("iters_before_timestep_restart",d_max_num_iterations);
  ps->appendElement("num_iters_to_decrease_delT",d_num_iters_to_decrease_delT);
  ps->appendElement("num_iters_to_increase_delT",d_num_iters_to_increase_delT);
  ps->appendElement("delT_decrease_factor",d_delT_decrease_factor);
  ps->appendElement("delT_increase_factor",d_delT_increase_factor);

  ps->appendElement("solver",d_solver_type);
  ps->appendElement("temperature_solve",d_temp_solve);
  ps->appendElement("interpolateParticleTempToGridEveryStep",
                  d_interpolateParticleTempToGridEveryStep);

}

