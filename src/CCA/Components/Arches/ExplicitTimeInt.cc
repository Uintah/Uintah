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
#include <Core/Grid/MaterialManager.h>
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
                                       std::vector<std::string> phi,
                                       std::vector<std::string> rhs, 
                                       int rkstep )
{
  
  Task* tsk = scinew Task("ExplicitTimeInt::fe_update", this, &ExplicitTimeInt::fe_update, phi, rhs, rkstep);
  Ghost::GhostType ghost_type = Ghost::None; 
  int n_extra = 0;
  
  // phi
  for ( std::vector<std::string>::iterator iter = phi.begin(); iter != phi.end(); iter++ ){ 

    const VarLabel* lab = VarLabel::find( *iter );
    
    tsk->modifiesVar( lab );
    tsk->requiresVar( Task::OldDW, lab, ghost_type, n_extra );
  } 
  // rhs
  for ( std::vector<std::string>::iterator iter = rhs.begin(); iter != rhs.end(); iter++ ){ 

    const VarLabel* lab = VarLabel::find( *iter ); 

    tsk->requiresVar( Task::NewDW, lab, ghost_type, n_extra ); 

  } 

  tsk->requiresVar(Task::OldDW, d_fieldLabels->d_delTLabel, Ghost::None, 0);

  sched->addTask( tsk, patches, matls ); 

}

void ExplicitTimeInt::fe_update( const ProcessorGroup*, 
                                 const PatchSubset* patches, 
                                 const MaterialSubset* matls, 
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw, 
                                 std::vector<std::string> phi_tag, 
                                 std::vector<std::string> rhs_tag, 
                                 int rkstep )
{ 
  int N = phi_tag.size(); 
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; 
    int indx = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
    Ghost::GhostType ghost_type = Ghost::None; 
    int n_extra = 0;
    
    for ( int i = 0; i < N; i++){ 

      CCVariable<double> phi; 
      constCCVariable<double> phi_old; 
      constCCVariable<double> rhs; 

      const VarLabel* phi_lab = VarLabel::find( phi_tag[i] ); 
      const VarLabel* rhs_lab = VarLabel::find( rhs_tag[i] ); 
      old_dw->get( phi_old           , phi_lab , indx , patch , ghost_type, n_extra  );
      
      new_dw->getModifiable( phi , phi_lab , indx , patch, ghost_type, n_extra  );

      new_dw->get( rhs               , rhs_lab , indx , patch , ghost_type, n_extra  );

      std::string eqn_name = "some_eqn"; 

      delt_vartype DT;
      old_dw->get(DT, d_fieldLabels->d_delTLabel);
      double dt = DT; 

      singlePatchFEUpdate( patch, 
                          phi, 
                          rhs, 
                          dt, 
                          eqn_name);
    } 
  }
} 

//---------------------------------------------------------------------------
// Method: Schedule a time average
//---------------------------------------------------------------------------
void ExplicitTimeInt::sched_time_ave( SchedulerP& sched, 
                                      const PatchSet* patches, 
                                      const MaterialSet* matls, 
                                      std::vector<std::string> phi,
                                      int rkstep)
{
  
  Task* tsk = scinew Task("ExplicitTimeInt::time_ave", this, &ExplicitTimeInt::time_ave, phi, rkstep);
  Ghost::GhostType ghost_type = Ghost::None; 
  int extra_cells = 0;

  for ( std::vector<std::string>::iterator iter = phi.begin(); iter != phi.end(); iter++ ){ 

    const VarLabel* lab = VarLabel::find( *iter ); 

    tsk->requiresVar( Task::OldDW, lab, ghost_type, extra_cells ); 
    tsk->modifiesVar( lab ); 

  } 

  sched->addTask( tsk, patches, matls ); 

}

void ExplicitTimeInt::time_ave( const ProcessorGroup*, 
                                const PatchSubset* patches, 
                                const MaterialSubset* matls, 
                                DataWarehouse* old_dw, 
                                DataWarehouse* new_dw, 
                                std::vector<std::string> phi_tag, 
                                int rkstep )
{ 
  Ghost::GhostType ghost_type = Ghost::None; 
  int extra_cells = 0;
  
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int archIndex = 0; 
    int indx = d_fieldLabels->d_materialManager->getMaterial( "Arches", archIndex)->getDWIndex(); 
    //Ghost::GhostType gn = Ghost::None; 

    for ( std::vector<std::string>::iterator i = phi_tag.begin(); i != phi_tag.end(); i++ ){ 

      CCVariable<double> phi; 
      constCCVariable<double> old_phi;

      const VarLabel* phi_lab = VarLabel::find( *i ); 

      new_dw->getModifiable( phi , phi_lab , indx , patch );
      old_dw->get( old_phi       , phi_lab , indx , patch , ghost_type , extra_cells );

      timeAvePhi( patch, phi, old_phi, rkstep ); 

    } 
  }
}  
