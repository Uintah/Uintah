#ifndef Uintah_Component_Arches_BowmanNOx_h
#define Uintah_Component_Arches_BowmanNOx_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

/** 
* @class  Bowman, 1975 NOx rate of formation
* @author Jeremy Thornock
* @date   Oct 2011
* 
* @brief Computes the rate term according to Bowman 1975 for NOx formation. 
*
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <Sources>
*     <src label = "my_source" type = "bowman_nox" > 
			 	<!-- Bowman NOx rate expression --> 
			 	<n2_label 								  spec="OPTIONAL STRING" need_applies_to="type bowman_nox"/> <!-- mass fraction label for N2 (default = N2) --> 
        <A                          spec="REQUIRED DOUBLE" need_applies_to="type bowman_nox"/> <!-- Pre-exponential factor --> 
        <E_R                        spec="REQUIRED DOUBLE" need_applies_to="type bowman_nox"/> <!-- Activation temperature, code multiplies the -1!! --> 
        <o2_label                   spec="OPTIONAL STRING" need_applies_to="type bowman_nox"/> <!-- o2 label (default = O2) --> 
        <temperature_label          spec="OPTIONAL STRING" need_applies_to="type bowman_nox"/> <!-- temperature label (default = temperature) --> 
        <density_label              spec="OPTIONAL STRING" need_applies_to="type bowman_nox"/> <!-- density label (default = "density) --> 
      </src>
    </Sources>
* \endcode 
*  
*/ 
namespace Uintah{

class BowmanNOx: public SourceTermBase {
public: 

  BowmanNOx( std::string srcName, ArchesLabel* field_labels, 
                vector<std::string> reqLabelNames, std::string type );

  ~BowmanNOx();
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
  /** @brief Schedule a dummy initialization */ 
  void sched_dummyInit( const LevelP& level, SchedulerP& sched );
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls, 
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, vector<std::string> required_label_names, ArchesLabel* field_labels ) 
        : _name(name), _field_labels(field_labels), _required_label_names(required_label_names){
          _type = "bowman_nox"; 
        };
      ~Builder(){}; 

      BowmanNOx* build()
      { return scinew BowmanNOx( _name, _field_labels, _required_label_names, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      ArchesLabel* _field_labels; 
      vector<std::string> _required_label_names; 

  }; // Builder


private:

  ArchesLabel* _field_labels;

  double _MW_N2;            ///< molecular weight of no2
  double _MW_O2;            ///< moleculat weight of o2
  double _A;                ///< pre-exponential factor [units?]
  double _E_R;              ///< activation temperature [K] POSITIVE VALUE! Code will multiply the negative

  std::string _n2_name;           ///< string name for no2 (from table)
  std::string _o2_name;           ///< string name for o2  (from table)
  std::string _rho_name;          ///< string name for rho (from table)
  std::string _temperature_name;  ///< string name for temperature (from table)

  const VarLabel* _n2_label;  ///< n2 label
  const VarLabel* _o2_label;  ///< o2  label 
  const VarLabel* _rho_label; ///< rho label 
  const VarLabel* _temperature_label; ///< temperature label 



}; // end BowmanNOx
} // end namespace Uintah
#endif
