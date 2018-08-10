#ifndef Uintah_Component_Arches_CoalGasMomentum_h
#define Uintah_Component_Arches_CoalGasMomentum_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
/**
 * @class  CoalGasMomentum
 * @author Jeremy Thornock, Julien Pedel, Charles Reid
 * @date   Not sure
 *
 * @brief Assembles source term for the particle drag from the
 *        particle phase. 
 *
 * @todo
 *
 * @details
 * This simply packages a gas source term into a form that is easily 
 * accessed by the moementum equation.  The packaging is an assembly of
 * the various model terms integrated over the ndf using the weights to 
 * provide a total enthalpy source term. 
 *
 * Input file interface is as follows: 
\code
<Sources>
  <src label="STRING REQUIRED" type="coal_gas_momentum"/>
</Sources>
\endcode
 *
*/

namespace Uintah{

class CoalGasMomentum: public SourceTermBase {
public: 

  CoalGasMomentum( std::string srcName, MaterialManagerP& materialManager, 
                std::vector<std::string> reqLabelNames, std::string type );

  ~CoalGasMomentum();
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
          _type = "coal_has_momentum"; 
        };
      ~Builder(){}; 

      CoalGasMomentum* build()
      { return scinew CoalGasMomentum( _name, _materialManager, _required_label_names, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      MaterialManagerP& _materialManager; 
      std::vector<std::string> _required_label_names;

  }; // Builder
private:

  //double d_constant; 
  std::string d_dragModelName;

}; // end CoalGasMomentum
} // end namespace Uintah
#endif
