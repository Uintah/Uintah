#ifndef Uintah_Component_Arches_PropertyModelBase_h
#define Uintah_Component_Arches_PropertyModelBase_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>

#include <typeinfo>


/**
* @class  PropertyModelBase
* @author Jeremy Thornock
* @date   Aug. 2011
*
* @brief A base class for property models.
*
*
*/

namespace Uintah {

class PropertyModelBase{

public:

  PropertyModelBase( std::string prop_name, SimulationStateP& shared_state );
  virtual ~PropertyModelBase();

  /** @brief Interface to the input file */
  virtual void problemSetup( const ProblemSpecP& db ) = 0;

  /** @brief Scheduler for the actual property calculation */
  virtual void sched_computeProp( const LevelP& level, SchedulerP& sched, int time_substep ) = 0;

  /** @brief Common setup for all models. **/
  void commonProblemSetup( const ProblemSpecP& inputdb );

  /** @brief Scheduler for the initialization of the property **/
  virtual void sched_initialize( const LevelP& level, SchedulerP& sched ) = 0;

  /** @brief Initialize memory for the property **/
  void sched_timeStepInit( const LevelP& level, SchedulerP& sched );
  void timeStepInit( const ProcessorGroup* pc,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw
                     );

  /** @brief Returns the property label */
  inline const VarLabel* getPropLabel(){
    return _prop_label; };

  /** @brief Returns the property type as set in the derived class **/
  inline const std::string getPropType(){
    return _prop_type; };

  /** @brief Returns the name of the property **/
  inline const std::string retrieve_property_name(){ return _prop_name; } 

  /** @brief Returns the initialization type as set in the derived class **/
  inline const std::string initType(){
    return _init_type; };

  /** @brief  Initialize variables during a restart */
  virtual void sched_restartInitialize( const LevelP& level, SchedulerP& sched ){};

  /** @brief Returns a vector of extra labels stored for this specific property **/
  inline const std::vector<const VarLabel*> getExtraLocalLabels(){
    return _extra_local_labels; };

  /** @brief Returns the boolean to indicate if the model is to be evaluated before or after the table lookup **/
  inline bool beforeTableLookUp() { return _before_table_lookup; };

  /** @brief Builder class containing instructions on how to build the property model **/
  class Builder {

    public:

      virtual ~Builder() {}

      virtual PropertyModelBase* build() = 0;

    protected:

      std::string _name;
  };


protected:

  std::string _prop_name;                             ///< User assigned property name
  std::vector<const VarLabel*> _extra_local_labels;   ///< Vector of extra local labels
  std::string _init_type;                             ///< Initialization type
  std::string _prop_type;                             ///< The actual type of property (heat_loss, soot, etc...)

  const VarLabel* _prop_label;                        ///< Property varlabel

  bool _before_table_lookup;                          ///< To determine if the property model is evaluated before the table look up or after.

  SimulationStateP& _shared_state;                    ///< Uintah shared state

  /** @brief A common intialization proceedure that can be used by all derived types */
  template <class phiT >
  void base_initialize( const Patch* patch, phiT& phi );


  // Constant initialization
  double _const_init;                                 ///< Constant for intialization

  // Gaussian initialization
  int _dir_gauss;
  double _a_gauss;
  double _b_gauss;
  double _c_gauss;
  double _shift_gauss;

}; // end PropertyModelBase

template <class phiT >
void PropertyModelBase::base_initialize( const Patch* patch, phiT& phi ){

  std::string msg = "Initializing property models. ";
  proc0cout << msg << std::endl;

  if ( _init_type == "constant" ) {

    phi.initialize( _const_init );

  } else if ( _init_type == "gaussian" ) {

    //======= Gaussian ========
    for (CellIterator iter=patch->getCellIterator(0); !iter.done(); iter++){

      IntVector c = *iter;
      Point  P  = patch->getCellPosition(c);

      double x=0.0,y=0.0,z=0.0;

      x = P.x();
      y = P.y();
      z = P.z();

      if ( _dir_gauss == 0 ){

        phi[c] = _a_gauss * exp( -1.0*std::pow(x-_b_gauss,2.0)/(2.0*std::pow(_c_gauss,2.0))) + _shift_gauss;

      } else if ( _dir_gauss == 1 ){

        phi[c] = _a_gauss * exp( -1.0*std::pow(y-_b_gauss,2.0)/(2.0*std::pow(_c_gauss,2.0))) + _shift_gauss;

      } else {

        phi[c] = _a_gauss * exp( -1.0*std::pow(z-_b_gauss,2.0)/(2.0*std::pow(_c_gauss,2.0))) + _shift_gauss;

      }
    }

  } else if (_init_type == "physical" ){
    _init_type = "physical";
  } else {
    proc0cout << " For property model: " << _prop_name << std::endl;
    throw InvalidValue("Initialization type for property model not recognized or supported!", __FILE__, __LINE__);
  }
}


}  // end namespace Uintah

#endif
