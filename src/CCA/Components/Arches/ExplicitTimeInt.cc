#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
//#include <Core/Grid/Task.h>
//#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>

//===========================================================================

using namespace Uintah;

ExplicitTimeInt::ExplicitTimeInt(const ArchesLabel* fieldLabels):
d_fieldLabels(fieldLabels)
{}

ExplicitTimeInt::~ExplicitTimeInt()
{}
//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void ExplicitTimeInt::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP ex_db = params->findBlock("ExplicitIntegrator");

  ex_db->getAttribute("order", d_time_order); 

  if (d_time_order == "first"){
    
    ssp_alpha[0] = 0.0;
    ssp_alpha[1] = 0.0;
    ssp_alpha[2] = 0.0;

    ssp_beta[0]  = 1.0;
    ssp_beta[1]  = 0.0;
    ssp_beta[2]  = 0.0;

    time_factor[0] = 1.0;
    time_factor[1] = 0.0;
    time_factor[2] = 0.0; 

  }
  else if (d_time_order == "second") {

    ssp_alpha[0]= 0.0;
    ssp_alpha[1]= 0.5;
    ssp_alpha[2]= 0.0;

    ssp_beta[0]  = 1.0;
    ssp_beta[1]  = 0.5;
    ssp_beta[2]  = 0.0;

    time_factor[0] = 1.0;
    time_factor[1] = 1.0;
    time_factor[2] = 0.0; 

  }
  else if (d_time_order == "third") {

    ssp_alpha[0] = 0.0;
    ssp_alpha[1] = 0.75;
    ssp_alpha[2] = 1.0/3.0;

    ssp_beta[0]  = 1.0;
    ssp_beta[1]  = 0.25;
    ssp_beta[2]  = 2.0/3.0;

    time_factor[0] = 1.0;
    time_factor[1] = 0.5;
    time_factor[2] = 1.0; 

  }
  else
            throw InvalidValue("Explicit time integration order must be one of: first, second, third!  Please fix input file.",__FILE__, __LINE__);             
}
//---------------------------------------------------------------------------
// Method: Schedule a time update
//---------------------------------------------------------------------------
void ExplicitTimeInt::sched_fe_update( SchedulerP& sched, 
                                       const PatchSet* patches, 
                                       const MaterialSet* matls, 
                                       std::vector<const VarLabel*> phi,
                                       std::vector<const VarLabel*> rhs, 
                                       int rkstep )
{
  
  Task* tsk = scinew Task("ExplicitTimeInt::fe_update", this, &ExplicitTimeInt::fe_update, phi, rhs, rkstep); 
  Ghost::GhostType gn = Ghost::None; 

  // phi
  for ( std::vector<const VarLabel*>::iterator iter = phi.begin(); iter != phi.end(); iter++ ){ 

    tsk->modifies( *iter ); 

  } 
  // rhs
  for ( std::vector<const VarLabel*>::iterator iter = rhs.begin(); iter != rhs.end(); iter++ ){ 

    tsk->requires( Task::NewDW, *iter, gn, 0 ); 

  } 

  sched->addTask( tsk, patches, matls ); 

}

void ExplicitTimeInt::fe_update( const ProcessorGroup*, 
                                 const PatchSubset* patches, 
                                 const MaterialSubset* matls, 
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw, 
                                 std::vector<const VarLabel*> phi_lab, 
                                 std::vector<const VarLabel*> rhs_lab, 
                                 int rkstep )
{ 
  int N = phi_lab.size(); 
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; 
    int indx = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 
    Ghost::GhostType gn = Ghost::None; 

    for ( int i = 0; i < N; i++){ 

      CCVariable<double> phi; 
      constCCVariable<double> rhs; 

      new_dw->getModifiable( phi, phi_lab[i], indx, patch ); 
      new_dw->get( rhs, rhs_lab[i], indx, patch, gn, 0 ); 

      std::string eqn_name = "some_eqn"; 

      delt_vartype DT;
      old_dw->get(DT, d_fieldLabels->d_sharedState->get_delt_label());
      double dt = DT; 

      double curr_time = d_fieldLabels->d_sharedState->getElapsedTime(); 
      double curr_ssp_time = curr_time + time_factor[rkstep] * dt;

      singlePatchFEUpdate( patch, 
                           phi, 
                           rhs, 
                           dt, curr_ssp_time, 
                           eqn_name ); 

    } 
  }
} 

