#ifndef Uintah_Component_Arches_HTConvection_h
#define Uintah_Component_Arches_HTConvection_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/ArchesVariables.h>

namespace Uintah{

class HTConvection: public SourceTermBase {
public: 

  HTConvection( std::string srcName, ArchesLabel* field_labels, 
                std::vector<std::string> reqLabelNames, std::string type );

  ~HTConvection();
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

      Builder( std::string name, std::vector<std::string> required_label_names, ArchesLabel* field_labels )
        : _name(name), _field_labels(field_labels), _required_label_names(required_label_names){
          _type = "HTConvection"; 
        };
      ~Builder(){}; 

      HTConvection* build()
      { return scinew HTConvection( _name, _field_labels, _required_label_names, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      ArchesLabel* _field_labels; 
      std::vector<std::string> _required_label_names;

  }; // Builder


private:

  ArchesLabel* _field_labels;
  std::string ConWallHT_src_name;
  double ThermalConductGas(double Tg, double Tp);
  const VarLabel* _volFraction_varlabel;
  const VarLabel* _gas_temperature_varlabel;


  const VarLabel* ConWallHT_src_label;
  
  double  _dTCorrectionFactor; /// The correction factor for the thermal layer
  std::string _gas_temperature_name;    ///< string name for temperature (from table)
    
}; // end HTConvection
} // end namespace Uintah
#endif
