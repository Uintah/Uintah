/** * @class NO prediction model
 * * @author Model developed by Zhi Zhang, Zhenshan Li, Ningsheng Cai from Tsinghua University;
 * * @brief Coding by Zhi Zhang under the instruction of Minmin Zhou, Ben Issac and Jeremy Thornock;
 *  Parameters fitted based on DTF experimental data of a Chinese bituminous coal Tsinghua.
 *  NO,HCN,NH3 transport equations and sources terms in both gas and solid phase.
 * */
#ifndef Uintah_Component_Arches_psNox_h
#define Uintah_Component_Arches_psNox_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Directives.h>

/**
* The input file interface for this property should like this in your UPS file:
* \code
*   <Sources>
*     <src label = "zz_Nox_solidphase_source" type = "psNox" >
        <!-- B Source Term -->
        <A_BET          spec="OPTIONAL STRING" need_applies_to="type psNox"/> <!-- parameter for zz model -->
        <NSbeta1        spec="REQUIRED DOUBLE" need_applies_to="type psNox"/> <!--  parameter for zz model-->
        <NSbeta2        spec="REQUIRED DOUBLE" need_applies_to="type psNox"/> <!-- parameter for zz model -->
        <NSgamma1       spec="OPTIONAL STRING" need_applies_to="type psNox"/> <!-- parameter for zz model-->
        <NSgamma2       spec="OPTIONAL STRING" need_applies_to="type psNox"/> <!--parameter for zz model -->

        <NO_src       spec="REQUIRED NO_DATA"
                      attribute1="label REQUIRED STRING"
                      need_applies_to="type psNox"/> <!-- SourceTerm for NO transport -->
        <HCN_src      spec="REQUIRED NO_DATA"
                      attribute1="label REQUIRED STRING"
                      need_applies_to="type psNox"/> <!-- SourceTerm for HCN transport-->
        <NH3_src      spec="REQUIRED NO_DATA"
                      attribute1="label REQUIRED STRING"
                      need_applies_to="type psNox"/> <!--SourceTerm for NH3-->
      </src>
    </Sources>
* \endcode
*
*/
namespace Uintah{

class psNox: public SourceTermBase {
public:

  psNox( std::string srcName, ArchesLabel* field_labels,
             std::vector<std::string> reqLabelNames, std::string type );

  ~psNox();
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
          _type = "psNox";
        };
      ~Builder(){};

      psNox* build()
      { return scinew psNox( _name, _field_labels, _required_label_names, _type ); };

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
    double _A_BET;
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
  std::string devol_name; ///< string name for the average molecular weight (from table)
  std::string bd_devol_name; 
  std::string oxi_name;            ///< string name for tar
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

  std::string NO_src_name; ///< string name for the average molecular weight (from table)
  std::string HCN_src_name;            ///< string name for tar
  std::string NH3_src_name;        ///< string name for tar src
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

/**
 NO sourceTerm computation
struct ComputNOSource{
       ComputNOSource(constCCVariable<double>& _devol,
                      constCCVariable<double>& _oxi,
                      CCVariable<double>& _NO_src) :
#ifdef UINTAH_ENABLE_KOKKOS
                           devol(_devol.getKokkosView()),
                           oxi(_oxi.getKokkosView()),
                           NO_src(_NO_src.getKokkosView())
#else
                           devol(_devol),
                           oxi(_oxi),
                           NO_src(_NO_src)
#endif
                           {  }

  void operator()(int i , int j, int k ) const {
    NO_src(i,j,k) = _Nit*devol(i,j,k)+oxi(i,j,k);
  }

  private:
#ifdef UINTAH_ENABLE_KOKKOS
   KokkosView3<const double> devol;
   KokkosView3<const double> oxi;
   KokkosView3<double>  NO_src;
#else
   constCCVariable<double>& devol;
   constCCVariable<double>& oxi;
   CCVariable<double>& NO_src;
#endif
};
**/



}; // end psNox
} // end namespace Uintah

#endif
