#ifndef Uintah_Component_Arches_CoalGasHeat_h
#define Uintah_Component_Arches_CoalGasHeat_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>

/**
 * @class  CoalGasHeat
 * @author Jeremy Thornock, Julien Pedel, Charles Reid
 * @date   Not sure
 *
 * @brief Assembles source term for the gas enthalpy equation from the 
 *        particle phase. 
 *
 * @todo
 *
 * @details
 * This simply packages a gas source term into a form that is easily 
 * accessed by the enthalpy equation.  The packaging is an assembly of
 * the various model terms integrated over the ndf using the weights to 
 * provide a total enthalpy source term. 
 *
 * Input file interface is as follows: 
\code
<Sources>
  <src label="STRING OPTIONAL" type="coal_gas_heat">
    <heat_model_name>STRING REQUIRED</heat_model_name>
  </src>
</Sources>
\endcode
  * where heat_model_name is the given name of the heat transfer model for the 
  * particle model as specified in the CoalModel section. 
*/

namespace Uintah{ 

class CoalGasHeat: public SourceTermBase {

  public: 

  CoalGasHeat( std::string src_name, std::vector<std::string> required_label_names, MaterialManagerP& materialManager, std::string type );

  ~CoalGasHeat();

  /** @brief Interface for the inputfile and set constants */ 
  void problemSetup(const ProblemSpecP& db);

  /** @brief Schedule the calculation of the source term */ 
  void sched_computeSource( const LevelP& level, SchedulerP& sched, 
                            int timeSubStep );

  /** @brief Actually compute the source term */ 
  void computeSource( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw, 
                      int timeSubStep );

  /** @brief Schedule initialization */ 
  void sched_initialize( const LevelP& level, SchedulerP& sched );
  void initialize( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw );

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, std::vector<std::string> required_label_names, MaterialManagerP& materialManager )
        : _name(name), _materialManager(materialManager), _required_label_names(required_label_names){
          _type = "coal_gas_heat"; 
        };
      ~Builder(){}; 

      CoalGasHeat* build()
      { return scinew CoalGasHeat( _name, _required_label_names, _materialManager, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      MaterialManagerP& _materialManager; 
      std::vector<std::string> _required_label_names;

  }; // class Builder 

private:

  std::string m_enthalpy_root;
  std::string m_temperature_root;
  double _Ha0;
  std::vector<double> _mass_ash_vec;
  std::string _heat_model_name; 
  bool m_dest_flag;// flag indicating whether or not deposition mass will be added to the gas phase. 

}; // end CoalGasHeat
} // end namespace Uintah
#endif
