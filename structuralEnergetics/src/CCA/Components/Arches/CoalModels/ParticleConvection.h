#ifndef Uintah_Component_Arches_ParticleConvection_h
#define Uintah_Component_Arches_ParticleConvection_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

namespace Uintah{

//---------------------------------------------------------------------------
// Builder

class ParticleConvectionBuilder: public ModelBuilder
{
public: 
  ParticleConvectionBuilder( const std::string          & modelName, 
                        const std::vector<std::string>  & reqICLabelNames,
                        const std::vector<std::string>  & reqScalarLabelNames,
                        ArchesLabel          * fieldLabels,
                        SimulationStateP           & sharedState,
                        int qn );
  ~ParticleConvectionBuilder(); 

  ModelBase* build(); 

private:

}; 

// End Builder
//---------------------------------------------------------------------------

class ParticleConvection: public ModelBase {
public: 

  ParticleConvection( std::string modelName, 
                 SimulationStateP& shared_state, 
                 ArchesLabel* fieldLabels,
                 std::vector<std::string> reqICLabelNames,
                 std::vector<std::string> reqScalarLabelNames,
                 int qn );

  ~ParticleConvection();

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db, int qn);

  /** @brief Schedule the initialization of special/local variables unique to model */
  void sched_initVars( const LevelP& level, SchedulerP& sched );

  /** @brief  Actually initialize special variables unique to model */
  void initVars( const ProcessorGroup * pc, 
                 const PatchSubset    * patches, 
                 const MaterialSubset * matls, 
                 DataWarehouse        * old_dw, 
                 DataWarehouse        * new_dw );

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeModel( const LevelP& level, 
                           SchedulerP& sched, 
                           int timeSubStep );

  /** @brief Actually compute the source term */ 
  void computeModel( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw, 
                     const int timeSubStep );

  inline std::string getType() {
    return "Constant"; }

private:


}; // end Class
} // end namespace Uintah
#endif

