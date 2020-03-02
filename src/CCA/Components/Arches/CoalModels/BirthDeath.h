#ifndef Uintah_Component_Arches_BirthDeath_h
#define Uintah_Component_Arches_BirthDeath_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/CoalModels/ModelBase.h>
#include <CCA/Components/Arches/CoalModels/CoalModelFactory.h>

#include <CCA/Components/Arches/ArchesVariables.h>

#include <vector>
#include <string>

//===========================================================================

namespace Uintah{

//---------------------------------------------------------------------------
// Builder
class BirthDeathBuilder: public ModelBuilder
{
public:
  BirthDeathBuilder( const std::string          & modelName,
                        const std::vector<std::string>  & reqICLabelNames,
                        const std::vector<std::string>  & reqScalarLabelNames,
                        ArchesLabel          * fieldLabels,
                        MaterialManagerP           & materialManager,
                        int qn );
  ~BirthDeathBuilder();

  ModelBase* build();

private:

};
// End Builder
//---------------------------------------------------------------------------

class BirthDeath: public ModelBase {
public:

  BirthDeath( std::string modelName,
                 MaterialManagerP& materialManager,
                 ArchesLabel* fieldLabels,
                 std::vector<std::string> reqICLabelNames,
                 std::vector<std::string> reqScalarLabelNames,
                 int qn );

  ~BirthDeath();

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
    return "birth"; }


private:

  bool _is_weight;
  bool _deposition;

  std::string _abscissa_name;

  const VarLabel* _abscissa_label;
  const VarLabel* _w_label;
  const VarLabel* _w_rhs_label;
  const VarLabel* _rate_depX_varlabel;
  const VarLabel* _rate_depY_varlabel;
  const VarLabel* _rate_depZ_varlabel;
  const VarLabel* _length_varlabel;
  const VarLabel* _particle_density_varlabel;

  const VarLabel* _rate_impactX_varlabel;
  const VarLabel* _rate_impactY_varlabel;
  const VarLabel* _rate_impactZ_varlabel;

  VarLabel* d_model_impactLabel;

  double _small_weight;
  double _a_scale;
  double _w_scale;
  double _pi;


}; // end ConstSrcTerm
} // end namespace Uintah
#endif
