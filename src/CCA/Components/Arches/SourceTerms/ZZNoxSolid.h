/** * @class NO prediction model
 * * @author Model developed by Zhi Zhang, Zhenshan Li, Ningsheng Cai from Tsinghua University;
 * * @brief Coding by Zhi Zhang under the instruction of Minmin Zhou, Ben Issac and Jeremy Thornock;
 *  Parameters fitted based on DTF experimental data of a Chinese bituminous coal Tsinghua.
 *  NO,HCN,NH3 transport equations and sources terms in both gas and solid phase.
 * */
#ifndef Uintah_Component_Arches_ZZNoxSolid_h
#define Uintah_Component_Arches_ZZNoxSolid_h

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
*     <src label = "zz_Nox_solidphase_source" type = "zzNoxSolid" >
        <!-- B Source Term -->
        <A_BET          spec="OPTIONAL STRING" need_applies_to="type ZZNoxSolid"/> <!-- parameter for zz model -->
        <NSbeta1        spec="REQUIRED DOUBLE" need_applies_to="type ZZNoxSolid"/> <!--  parameter for zz model-->
        <NSbeta2        spec="REQUIRED DOUBLE" need_applies_to="type ZZNoxSolid"/> <!-- parameter for zz model -->
        <NSgamma1       spec="OPTIONAL STRING" need_applies_to="type ZZNoxSolid"/> <!-- parameter for zz model-->
        <NSgamma2       spec="OPTIONAL STRING" need_applies_to="type ZZNoxSolid"/> <!--parameter for zz model -->
     
        <NO_src       spec="REQUIRED NO_DATA"
                      attribute1="label REQUIRED STRING"
                      need_applies_to="type ZZNoxSolid"/> <!-- SourceTerm for NO transport -->
        <HCN_src      spec="REQUIRED NO_DATA"
                      attribute1="label REQUIRED STRING"
                      need_applies_to="type ZZNoxSolid"/> <!-- SourceTerm for HCN transport-->
        <NH3_src      spec="REQUIRED NO_DATA"
                      attribute1="label REQUIRED STRING"
                      need_applies_to="type ZZNoxSolid"/> <!--SourceTerm for NH3-->
      </src>
    </Sources>
* \endcode
*
*/
namespace Uintah{

class ZZNoxSolid: public SourceTermBase {
public:

  ZZNoxSolid( std::string srcName, ArchesLabel* field_labels,
             std::vector<std::string> reqLabelNames, std::string type );

  ~ZZNoxSolid();
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
          _type = "ZZNoxSolid";
        };
      ~Builder(){};

      ZZNoxSolid* build()
      { return scinew ZZNoxSolid( _name, _field_labels, _required_label_names, _type ); };

    private:

      std::string _name;
      std::string _type;
      ArchesLabel* _field_labels;
      std::vector<std::string> _required_label_names;

  }; // Builder


private:

  ArchesLabel* _field_labels;

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
    int m_num_env; 

  std::string devol_name; ///< string name for the average molecular weight (from table)
  std::string oxi_name;            ///< string name for tar
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
  std::string m_rcmass_root; 
  std::string m_rho_coal_root; 
  std::string m_coal_temperature_root; 
  std::vector<double> m_rc_scaling_const; 
  std::vector<double> m_weight_scaling_const; 
  std::vector<double> m_particle_size; 

  const VarLabel* NO_src_label;
  const VarLabel* HCN_src_label;
  const VarLabel* NH3_src_label;
  const VarLabel* devol_label;
  const VarLabel* oxi_label;
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



}; // end ZZNoxSolid
} // end namespace Uintah

#endif
