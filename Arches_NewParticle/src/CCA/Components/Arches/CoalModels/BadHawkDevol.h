#ifndef Uintah_Component_Arches_BadHawkDevol_h
#define Uintah_Component_Arches_BadHawkDevol_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>
#include <CCA/Components/Arches/ArchesVariables.h>

namespace Uintah{

//===========================================================================

/**
  * @class    BadHawkDevol
  * @author   Charles Reid
  * @date     October 2009
  *
  * @brief    A class for calculating the DQMOM model term for the 
  *           Badzioch and Hawksley coal devolatilization model.
  *
  */

//---------------------------------------------------------------------------
// Builder
class BadHawkDevolBuilder: public ModelBuilder
{
public: 
  BadHawkDevolBuilder( const std::string          & modelName,
                           const vector<std::string>  & reqICLabelNames,
                           const vector<std::string>  & reqScalarLabelNames,
                           const ArchesLabel          * fieldLabels,
                           SimulationStateP           & sharedState,
                           int qn );

  ~BadHawkDevolBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class BadHawkDevol: public Devolatilization {
public: 

  BadHawkDevol( std::string modelName, 
                SimulationStateP& shared_state, 
                const ArchesLabel* fieldLabels,
                vector<std::string> reqICLabelNames, 
                vector<std::string> reqScalarLabelNames,
                int qn );

  ~BadHawkDevol();

  ////////////////////////////////////////////////
  // Initialization/setup methods

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  ////////////////////////////////////////////////
  // Model computation methods

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

private:

  const ArchesLabel* d_fieldLabels; 
  
  double A1;
  double A2;
  
  double E1;
  double E2;
  
  double R;
  
  bool compute_part_temp;

  double c_o;      // initial mass of raw coal
  double alpha_o;  // initial mass fraction of raw coal

}; // end ConstSrcTerm
} // end namespace Uintah
#endif
