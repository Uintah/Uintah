#ifndef Uintah_Component_Arches_SimpleBirth_h
#define Uintah_Component_Arches_SimpleBirth_h
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
class SimpleBirthBuilder: public ModelBuilder
{
public: 
  SimpleBirthBuilder( const std::string          & modelName, 
                        const std::vector<std::string>  & reqICLabelNames,
                        const std::vector<std::string>  & reqScalarLabelNames,
                        ArchesLabel          * fieldLabels,
                        SimulationStateP           & sharedState,
                        int qn );
  ~SimpleBirthBuilder(); 

  ModelBase* build(); 

private:

}; 
// End Builder
//---------------------------------------------------------------------------

class SimpleBirth: public ModelBase {
public: 

  SimpleBirth( std::string modelName, 
                 SimulationStateP& shared_state, 
                 ArchesLabel* fieldLabels,
                 std::vector<std::string> reqICLabelNames,
                 std::vector<std::string> reqScalarLabelNames,
                 int qn );

  ~SimpleBirth();

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

  bool _is_weight; 

  std::string _abscissa_name; 

  const VarLabel* _abscissa_label; 
  const VarLabel* _w_label; 
  const VarLabel* _w_rhs_label;

  double _small_weight; 
  double _a_scale; 


}; // end ConstSrcTerm
} // end namespace Uintah
#endif

