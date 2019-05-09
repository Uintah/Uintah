#ifndef Uintah_Component_Arches_psNox_h
#define Uintah_Component_Arches_psNox_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

namespace Uintah{

class psNOx: public SourceTermBase {
public:

  psNOx( std::string srcName, ArchesLabel* field_labels,
             std::vector<std::string> reqLabelNames, std::string type );

  ~psNOx();
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
          _type = "psNOx";
        };
      ~Builder(){};

      psNOx* build()
      { return scinew psNOx( _name, _field_labels, _required_label_names, _type ); };

    private:

      std::string _name;
      std::string _type;
      ArchesLabel* _field_labels;
      std::vector<std::string> _required_label_names;

  }; // Builder


private:

  ArchesLabel* _field_labels;

    double m_v_hiT;
    double _Nit;
    double _N_ad;
    double _Ash_ad;
    double _H2O_ad;
    double _gasPressure;
    double _beta1;
    double _beta2;
    double _beta3;
    double _gamma1;
    double _gamma2;
    double _gamma3;
    double _alpha1;
    double _alpha2;
    double _alpha3;
    double _A_reburn;
    double _E_reburn;
    double _m_gr;
    double _F1_Desoete;

    double tarFrac;
    double m_ash_mass_fraction;
    int m_num_env;

  std::string tar_src_name;
  std::string devol_name;            ///< string name for the average molecular weight (from table)
  std::string bd_devol_name;
  std::string oxi_name;              ///< string name for tar
  std::string bd_oxi_name;
  std::string m_O2_name;
  std::string m_N2_name;
  std::string m_CO_name;
  std::string m_H2O_name;
  std::string m_H2_name;
  std::string m_temperature_name;
  std::string m_density_name;
  std::string m_mix_mol_weight_name;
  std::string NO_name;
  std::string HCN_name;
  std::string NH3_name;

  std::string NO_src_name;              ///< string name for the average molecular weight (from table)
  std::string HCN_src_name;             ///< string name for tar
  std::string NH3_src_name;             ///< string name for tar src
  std::string m_coal_temperature_root;
  std::string m_weight_root;
  std::string m_length_root;
  std::string m_p_rho_root;
  std::string m_rc_mass_root;
  std::string m_char_mass_root;
  std::vector<double > m_initial_rc; // kg_i/#
  double m_Fd_M;
  double m_Fd_B;

  const VarLabel* NO_src_label;
  const VarLabel* HCN_src_label;
  const VarLabel* NH3_src_label;
  const VarLabel* tar_src_label;
  const VarLabel* devol_label;
  const VarLabel* oxi_label;
  const VarLabel* bd_devol_label;
  const VarLabel* bd_oxi_label;
  const VarLabel* m_o2_label;
  const VarLabel* m_n2_label;
  const VarLabel* m_co_label;
  const VarLabel* m_h2o_label;
  const VarLabel* m_h2_label;
  const VarLabel* m_temperature_label;
  const VarLabel* m_density_label;
  const VarLabel* m_mix_mol_weight_label;
  const VarLabel* m_NO_label;
  const VarLabel* m_HCN_label;
  const VarLabel* m_NH3_label;
  const VarLabel* m_NO_RHS_label;
  const VarLabel* m_HCN_RHS_label;
  const VarLabel* m_NH3_RHS_label;
  std::vector<const VarLabel* > m_coal_temperature_label;
  std::vector<const VarLabel* > m_weight_label;
  std::vector<const VarLabel* > m_length_label;
  std::vector<const VarLabel* > m_p_rho_label;
  std::vector<const VarLabel* > m_rc_mass_label;
  std::vector<const VarLabel* > m_char_mass_label;


}; // end psNOx
} // end namespace Uintah

#endif
