#ifndef Uintah_Component_Arches_BrownSootFormation_Tar_h
#define Uintah_Component_Arches_BrownSootFormation_Tar_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

/** 
* @class  Brown, 1998 Modeling Soot Derived from Pulverized Coal
* @author Alexander Josephson
* @date   Aug 2014
* 
* @brief Computes the soot formation source term from tar in a coal flame
 
* Alex Brown and Tom Fletcher, Energy and Fuels, Vol 12, No 4 1998, 745-757
*    Note, Alex used "c" to denote soot, here using "s" to denote soot. "t"
*        denotes tar.
*    Note, Alex's paper has a number of typos/units problems.
*    Reading Lee et al 1962 for soot oxidation, and Ma's dissertation (p 115 (102))
*    Alex's code is in his dissertation: the soot formation rate has [c_t], which is rhos*massFractionTar, not concentration as mol/m3, which is what is in his notation.  Also, Ma's Dissertation has afs_ = 5.02E8 1/s, implying the reaction as below.
* The input file interface for this property should like this in your UPS file: 
* \code 
*   <Sources>
*     <src label = "my_source" type = "BrownSootFromation_Tar" >
			 	<!-- Brown Soot Source Term -->
        <tar_label 					spec="OPTIONAL STRING" need_applies_to="type brown_soot"/> <!-- mass fraction label for tar (default = Tar) -->
        <A                          spec="REQUIRED DOUBLE" need_applies_to="type brown_soot"/> <!-- Pre-exponential factor --> 
        <E_R                        spec="REQUIRED DOUBLE" need_applies_to="type brown_soot"/> <!-- Activation temperature, code multiplies the -1!! --> 
        <o2_label                   spec="OPTIONAL STRING" need_applies_to="type brown_soot"/> <!-- o2 label (default = O2) --> 
        <temperature_label          spec="OPTIONAL STRING" need_applies_to="type brown_soot"/> <!-- temperature label (default = temperature) --> 
        <density_label              spec="OPTIONAL STRING" need_applies_to="type brown_soot"/> <!-- density label (default = "density) --> 
      </src>
    </Sources>
* \endcode 
*  
*/ 
namespace Uintah{

class BrownSootFormation_Tar: public SourceTermBase {
public: 

  BrownSootFormation_Tar( std::string srcName, ArchesLabel* field_labels, 
                std::vector<std::string> reqLabelNames, std::string type );

  ~BrownSootFormation_Tar();
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
          _type = "BrownSootFormation_Tar"; 
        };
      ~Builder(){}; 

      BrownSootFormation_Tar* build()
      { return new BrownSootFormation_Tar( _name, _field_labels, _required_label_names, _type ); };

    private: 

      std::string _name; 
      std::string _type; 
      ArchesLabel* _field_labels; 
      std::vector<std::string> _required_label_names;

  }; // Builder


private:

  ArchesLabel* _field_labels;

  
  std::string _mix_mol_weight_name; ///< string name for the average molecular weight (from table)
  std::string _tar_name;            ///< string name for tar (from table)
  std::string _Ysoot_name;          ///< string name for Ysoot
  std::string _Ns_name;             ///< string name for Ns (#/kg)
  std::string _o2_name;             ///< string name for o2  (from table)
  std::string _rho_name;            ///< string name for rho (from table)
  std::string _temperature_name;    ///< string name for temperature (from table)
  std::string _pressure_name;       ///< string name for pressure (from table)

void coalSootRR(const double P,
                const double T,
                const double rhoYO2,
                const double rhoYt,
                      double &Ytar_source
               );
    
    
}; // end BrownSootFormation_Tar
} // end namespace Uintah
#endif
