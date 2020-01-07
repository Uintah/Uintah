#ifndef Uintah_Component_Arches_MoMICSoot_h
#define Uintah_Component_Arches_MoMICSoot_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

/**
* @class  Josephson, 2018 Modeling Soot Formation from Solid Complex Fuels
* @author Alexander Josephson
* @date   September 2017
*
* @brief Computes the soot formation and evolution from precursors in a solid fuel flame using a sectional distribution for the precursors and method of moments with interpolative closure for the soot particles. Expensive model, use with extreme caution!!!
*
* Alexander Josephson, Rod Linn, and David Lignell, Combustion and Flame, 2018, 196, 265-283
*
* The input file interface for this property should like this in your UPS file:
* \code
*   <Sources>
*     <src label = "my_source" type = "momic_soot" >
        <!-- MoMIC Soot Source Term -->
        <PAH0_label                 spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- number density of soot precursors size of PAH0 (200 g/mole)>
        <PAH1_label                 spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- number density of soot precursors size of PAH1 (350 g/mole)>
        <PAH2_label                 spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- number density of soot precursors size of PAH2 (500 g/mole)>
        <PAH3_label                 spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- number density of soot precursors size of PAH3 (850 g/mole)>
        <PAH4_label                 spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- number density of soot precursors size of PAH4 (1200 g/mole)>
        <Msoot0_label               spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- zeroth moment of the soot particle size distribution (#/m3)>
        <Msoot1_label               spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- first moment of the soot particle size distribution (kg/m3)>
        <Msoot2_label               spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- second moment of the soot particle size distribution (kg2/m3)>
        <Msoot3_label               spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- third moment of the soot particle size distribution (kg3/m3)>
        <Msoot4_label               spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- fourth moment of the soot particle size distribution (kg4/m3)>
        <Msoot5_label               spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- fifth moment of the soot particle size distribution (#/m3)>
        <Msurface_label             spec="OPTIONAL STRING" need_applies_to="type momic_soot" /> <!-- dth moment of the soot particle size distribution (kg^d/m3)>
        <o2_label                   spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = O2) -->
        <co2_label                  spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = CO2) -->
        <oh_label                   spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = OH) -->
        <h2o_label                  spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = H2O) -->
        <pyrene_label               spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = PYRENE) -->
        <h2_label                   spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = H2) -->
        <h_label                    spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = H) -->
        <c2h2_label                 spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- o2 label (default = C2H2) -->
        <PAH0_src label="PAH0src"/>
        <PAH1_src label="PAH1src"/>
        <PAH2_src label="PAH2src"/>
        <PAH3_src label="PAH3src"/>
        <PAH4_src label="PAH4src"/>
        <Msoot0_src label="Msoot0src"/>
        <Msoot1_src label="Msoot1src"/>
        <Msoot2_src label="Msoot2src"/>
        <Msoot3_src label="Msoot3src"/>
        <Msoot4_src label="Msoot4src"/>
        <Msoot5_src label="Msoot5src"/>
        <Msurface_src label="Msurfsrc"/>
        <temperature_label          spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- temperature label (default = temperature) -->
        <density_label              spec="OPTIONAL STRING" need_applies_to="type momic_soot"/> <!-- density label (default = "density) -->
        <PAH0_src                   spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for PAH0 src
                                                                        (as generated by this model) -->
        <PAH1_src                   spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for PAH1 src
                                                                        (as generated by this model) -->
        <PAH2_src                   spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for PAH2 src
                                                                        (as generated by this model) -->
        <PAH3_src                   spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for PAH3 src
                                                                        (as generated by this model) -->
        <PAH4_src                   spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for PAH4 src
                                                                        (as generated by this model) -->
        <PAH5_src                   spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for PAH5 src
                                                                        (as generated by this model) -->
        <Msoot0_src                 spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for Msoot0 src
                                                                        (as generated by this model) -->
        <Msoot1_src                 spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for Msoot1 src
                                                                        (as generated by this model) -->
        <Msoot2_src                 spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for Msoot2 src
                                                                        (as generated by this model) -->
        <Msoot3_src                 spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for Msoot3 src
                                                                        (as generated by this model) -->
        <Msoot4_src                 spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for Msoot4 src
                                                                        (as generated by this model) -->
        <Msoot5_src                 spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for Msoot5 src
                                                                        (as generated by this model) -->
        <mass_balance_src           spec="REQUIRED NO_DATA"
                                    attribute1="label REQUIRED STRING"
                                    need_applies_to="type momic_soot"/> <!-- User defined label for the soot mass balance src
                                                                        (as generated by this model) -->
      </src>
    </Sources>
* \endcode
*
*/
namespace Uintah{

class MoMICSoot: public SourceTermBase {
public:

  MoMICSoot( std::string srcName, ArchesLabel* field_labels,
             std::vector<std::string> reqLabelNames, std::string type );

  ~MoMICSoot();
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
          _type = "MoMICSoot";
        };
      ~Builder(){};

      MoMICSoot* build()
      { return scinew MoMICSoot( _name, _field_labels, _required_label_names, _type ); };

    private:

      std::string _name;
      std::string _type;
      ArchesLabel* _field_labels;
      std::vector<std::string> _required_label_names;

  }; // Builder


private:

  ArchesLabel* _field_labels;

  double m_sys_pressure;

  std::string m_balance_name;
  std::string m_mix_mol_weight_name; ///< string name for the average molecular weight (from table)
  std::string m_Msoot0_name;         ///< string name for soot moments
  std::string m_Msoot0_src_name;     ///< string name for soot moments src
  std::string m_Msoot1_name;         ///< string name for soot moments
  std::string m_Msoot1_src_name;     ///< string name for soot moments src
  std::string m_Msoot2_name;         ///< string name for soot moments
  std::string m_Msoot2_src_name;     ///< string name for soot moments src
  std::string m_Msoot3_name;         ///< string name for soot moments
  std::string m_Msoot3_src_name;     ///< string name for soot moments src
  std::string m_Msoot4_name;         ///< string name for soot moments
  std::string m_Msoot4_src_name;     ///< string name for soot moments src
  std::string m_Msoot5_name;         ///< string name for soot moments
  std::string m_Msoot5_src_name;     ///< string name for soot moments src
  std::string m_Msurface_name;       ///< string name for soot surface moment
  std::string m_Msurface_src_name;   ///< string name for soot surface moment src
  std::string m_PAH0_name;           ///< string name for precursor sections
  std::string m_PAH0_src_name;       ///< string name for precursor sections src
  std::string m_PAH1_name;           ///< string name for precursor sections
  std::string m_PAH1_src_name;       ///< string name for precursor sections src
  std::string m_PAH2_name;           ///< string name for precursor sections
  std::string m_PAH2_src_name;       ///< string name for precursor sections src
  std::string m_PAH3_name;           ///< string name for precursor sections
  std::string m_PAH3_src_name;       ///< string name for precursor sections src
  std::string m_PAH4_name;           ///< string name for precursor sections
  std::string m_PAH4_src_name;       ///< string name for precursor sections src
  std::string m_O2_name;             ///< string name for o2  (from table)
  std::string m_OH_name;             ///< string name for oh  (from table)
  std::string m_CO2_name;            ///< string name for co2  (from table)
  std::string m_H2O_name;            ///< string name for h2o (from table)
  std::string m_H_name;              ///< string name for h (from table)
  std::string m_H2_name;             ///< string name for h2 (from table)
  std::string m_C2H2_name;           ///< string name for c2h2 (from table)
  std::string m_rho_name;            ///< string name for rho (from table)
  std::string m_pyrene_name;         ///< string name for pyrene (from table)
  std::string m_temperature_name;    ///< string name for temperature (from table)

  const VarLabel* m_Msoot0_src_label;
  const VarLabel* m_Msoot1_src_label;
  const VarLabel* m_Msoot2_src_label;
  const VarLabel* m_Msoot3_src_label;
  const VarLabel* m_Msoot4_src_label;
  const VarLabel* m_Msoot5_src_label;
  const VarLabel* m_Msurface_src_label;
  const VarLabel* m_PAH0_src_label;
  const VarLabel* m_PAH1_src_label;
  const VarLabel* m_PAH2_src_label;
  const VarLabel* m_PAH3_src_label;
  const VarLabel* m_PAH4_src_label;
  const VarLabel* m_balance_src_label;
  const VarLabel* m_mix_mol_weight_label;
  const VarLabel* m_Msoot0_label;
  const VarLabel* m_Msoot1_label;
  const VarLabel* m_Msoot2_label;
  const VarLabel* m_Msoot3_label;
  const VarLabel* m_Msoot4_label;
  const VarLabel* m_Msoot5_label;
  const VarLabel* m_Msurface_label;
  const VarLabel* m_PAH0_label;
  const VarLabel* m_PAH1_label;
  const VarLabel* m_PAH2_label;
  const VarLabel* m_PAH3_label;
  const VarLabel* m_PAH4_label;
  const VarLabel* m_o2_label;
  const VarLabel* m_oh_label;
  const VarLabel* m_co2_label;
  const VarLabel* m_h2o_label;
  const VarLabel* m_h_label;
  const VarLabel* m_h2_label;
  const VarLabel* m_c2h2_label;
  const VarLabel* m_pyrene_label;
  const VarLabel* m_temperature_label;
  const VarLabel* m_rho_label;

}; // end MoMICSoot
} // end namespace Uintah

#endif
