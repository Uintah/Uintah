#ifndef Uintah_Component_Arches_BadHawkDevol_h
#define Uintah_Component_Arches_BadHawkDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

/**
  * @class    BadHawkDevol
  * @author   Charles Reid
  * @date     October 2009
  *
  * @brief    A class for calculating the DQMOM model term for the 
  *           Badzioch and Hawksley coal devolatilization model.
  *
  * The Builder is required because of the Model Factory; the Factory needs
  * some way to create the model term and register it.
  *
  */

//---------------------------------------------------------------------------
// Builder
namespace Uintah{
class BadHawkDevolBuilder: public ModelBuilder
{
public: 
  BadHawkDevolBuilder( const std::string          & modelName,
                           const vector<std::string>  & reqICLabelNames,
                           const vector<std::string>  & reqScalarLabelNames,
                           ArchesLabel          * fieldLabels,
                           SimulationStateP           & sharedState,
                           int qn );

  ~BadHawkDevolBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class BadHawkDevol: public ModelBase {
public: 

  BadHawkDevol( std::string modelName, 
                    SimulationStateP& shared_state, 
                    ArchesLabel* fieldLabels,
                    vector<std::string> reqICLabelNames, 
                    vector<std::string> reqScalarLabelNames,
                    int qn );

  ~BadHawkDevol();

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );
  
  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw );

  /** @brief  Schedule the initialization of some special/local vars */ 
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize some special/local vars */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief  Schedule the dummy solve for MPMArches - see ExplicitSolver::noSolve */
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually do dummy solve */
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

private:
  
  map<string, string> LabelToRoleMap;

  const VarLabel* d_raw_coal_mass_label;
  const VarLabel* d_weight_label;
  const VarLabel* d_particle_temperature_label;

  double A1;
  double A2;
  
  double E1;
  double E2;
  
  double R;
  
  bool compute_part_temp;

  double c_o;      // initial mass of raw coal
  double alpha_o;  // initial mass fraction of raw coal

  int d_quad_node;   // store which quad node this model is for

  double d_rc_scaling_factor;
  double d_pt_scaling_factor;
  double d_w_scaling_factor; 
  double d_w_small; // "small" clip value for zero weights

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
