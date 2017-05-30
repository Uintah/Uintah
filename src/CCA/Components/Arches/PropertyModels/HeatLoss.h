#ifndef Uintah_Component_Arches_HeatLoss_h
#define Uintah_Component_Arches_HeatLoss_h
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>


// SEE PROPTEMPLATE.CC FOR INSTRUCTIONS

/**
* @class  HeatLoss
* @author Jeremy Thornock
* @date   Oct, 2012
*
* @brief Computes the heat loss for table look up.
*
* ADD INPUT FILE INFORMATION HERE:
* The input file interface for this property should like this in your UPS file:
* \code
*   <PropertyModels>
*     <.......>
*   </PropertyModels>
* \endcode
*
*/

namespace Uintah{

class BoundaryCondition_new;

class HeatLoss : public PropertyModelBase {

public:

  HeatLoss( std::string prop_name, SimulationStateP& shared_state );
  ~HeatLoss();

  void problemSetup( const ProblemSpecP& db );

  void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep );
  void computeProp( const ProcessorGroup * pc,
                    const PatchSubset    * patches,
                    const MaterialSubset * matls,
                    DataWarehouse        * old_dw,
                    DataWarehouse        * new_dw,
                    int                    time_substep );

  void sched_initialize( const LevelP& level, SchedulerP& sched );
  void initialize( const ProcessorGroup * pc,
                   const PatchSubset    * patches,
                   const MaterialSubset * matls,
                   DataWarehouse        * old_dw,
                   DataWarehouse        * new_dw );


  /** @brief get the bounds on heat loss from another source (table) **/
  void set_hl_bounds( std::vector<double> bounds ){
    _low_hl  = bounds[0];
    _high_hl = bounds[1];
  };

  std::string get_hs_label_name(){ return _sen_h_label_name; }
  std::string get_ha_label_name(){ return _adiab_h_label_name; }
  std::string get_enthalpy_name(){ return _enthalpy_label_name; }

  void sched_restartInitialize( const LevelP& level, SchedulerP& sched );
  void restartInitialize( const ProcessorGroup * pc,
                          const PatchSubset    * patches,
                          const MaterialSubset * matls,
                          DataWarehouse        * old_dw,
                          DataWarehouse        * new_dw );

  /** @brief get a reference to the table **/
  void set_table_ref( MixingRxnModel* rxn_model ){ _rxn_model = rxn_model; };

  class Builder : public PropertyModelBase::Builder {

  public:

    Builder( std::string name, SimulationStateP& shared_state ) : _name(name), _shared_state(shared_state) {};
    ~Builder(){};

    HeatLoss* build() { return scinew HeatLoss( _name, _shared_state ); };

  private:

    std::string _name;
    SimulationStateP& _shared_state;

  }; // class Builder

private:

  std::string _enthalpy_label_name;
  std::string _sen_h_label_name;
  std::string _adiab_h_label_name;

  BoundaryCondition_new* _boundary_condition;
  MixingRxnModel* _rxn_model;
  MixingRxnModel::InertMasterMap _inert_map;

  const VarLabel* _enthalpy_label;
  const VarLabel* _sen_h_label;
  const VarLabel* _adiab_h_label;
  const VarLabel* _vol_frac_label;

  double _low_hl;
  double _high_hl;
  double _constant;

  bool _noisy_heat_loss;
  bool _constant_heat_loss;
  bool _use_h_ad_lookup;

  const VarLabel* _actual_hl_label;              ///< If computing heat loss but not actually using it, then stuff the computed value of heat loss in here.

}; // class HeatLoss

} // end namespace Uintah

#endif
