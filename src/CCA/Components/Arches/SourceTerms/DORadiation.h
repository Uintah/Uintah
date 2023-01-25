#ifndef Uintah_Component_Arches_DORadiation_h
#define Uintah_Component_Arches_DORadiation_h
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>

/**
* @class  DORadiation
* @author Jeremy Thornock
* @date   August 2011
*
* @brief Computes the divergence of heat flux contribution from the
*         solution of the intensity equation.
*
* The input file interface for this property should like this in your UPS file:
*
*  <calc_frequency               spec="OPTIONAL INTEGER" need_applies_to="type do_radiation" /> <!-- calculate radiation every N steps, default = 3 -->
*  <calc_on_all_RKsteps          spec="OPTIONAL BOOLEAN" need_applies_to="type do_radiation" /> <!-- calculate radiation every RK step, default = false -->
*  <co2_label                    spec="OPTIONAL STRING"  need_applies_to="type do_radiation" /> <!-- string label with default of CO2, default = CO2 -->
*  <h2o_label                    spec="OPTIONAL STRING"  need_applies_to="type do_radiation" /> <!-- string label wtih default of H2O, default = H2O -->
*  <DORadiationModel             spec="REQUIRED NO_DATA" need_applies_to="type do_radiation" >
*    <opl                        spec="REQUIRED DOUBLE" />
*    <ordinates                  spec="OPTIONAL INTEGER" />
*    <property_model             spec="OPTIONAL STRING 'radcoef, patchmean, wsggm'" />
*    <LinearSolver               spec="OPTIONAL NO_DATA"
*                                     attribute1="type REQUIRED STRING 'hypre, petsc'">
*      <res_tol                  spec="REQUIRED DOUBLE" />
*      <ksptype                  spec="REQUIRED STRING 'gmres, cg'" />
*      <pctype                   spec="REQUIRED STRING 'jacobi, blockjacobi'" />
*      <max_iter                 spec="REQUIRED INTEGER" />
*    </LinearSolver>
*  </DORadiationModel>
*
* TO DO'S:
*
*/

namespace Uintah{

  class DORadiationModel;
  class ArchesLabel;
  class ApplicationInterface;

class DORadiation: public SourceTermBase {
public:

  DORadiation( std::string srcName, ArchesLabel* labels, MPMArchesLabel* MAlab,
               std::vector<std::string> reqLabelNames, const ProcessorGroup* my_world,
               std::string type );

  ~DORadiation();

  void problemSetup(const ProblemSpecP& db);
  void sched_computeSource( const LevelP& level, SchedulerP& sched,
                            int timeSubStep );

  void sched_computeSourceSweep( const LevelP& level, SchedulerP& sched,
                                 int timeSubStep );

  void computeSource( const ProcessorGroup* pc,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      int timeSubStep );

  void sched_initialize( const LevelP& level, SchedulerP& sched );
  void initialize( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  void sched_restartInitialize( const LevelP& level, SchedulerP& sched );
  void restartInitialize( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw );

//-------- Functions relevant to sweeps ----//
void init_all_intensities( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw );

// chains requires->modifies tasks to facilitate communication.   spatial scheduling for task work, but not communication
void doSweepAdvanced(  const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw ,
                         int intensity_iter );

// computes fluxes and divQ and volQ, by integrating intensities over the solid angle
void computeFluxDivQ( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw );

// profile radiaiton dynamics, to help users resolve radiation fields.
void profileDynamicRadiation( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw );

// initialize and set boundary conditions for intensities
void setIntensityBC( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         int ix );


void TransferRadFieldsFromOldDW( const ProcessorGroup* pc,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw);

void checkReductionVars( const ProcessorGroup * pg,
                         const PatchSubset    * patches,
                         const MaterialSubset * matls,
                               DataWarehouse  * old_dw,
                               DataWarehouse  * new_dw );

//---End of Functions relevant to sweeps ----//
  class Builder
    : public SourceTermBase::Builder {

    public:

      Builder( std::string name, std::vector<std::string> required_label_names, ArchesLabel* labels,
               const ProcessorGroup* my_world )
               : _name(name), _labels(labels),
                 _my_world(my_world), _required_label_names(required_label_names)
      {
        _type = "do_radiation";
      }

      ~Builder(){}

      DORadiation* build()
      { return scinew DORadiation( _name, _labels, _MAlab, _required_label_names, _my_world, _type ); }

    private:

      std::string _name;
      std::string _type;
      ArchesLabel* _labels;
      MPMArchesLabel* _MAlab;
      const ProcessorGroup* _my_world;
      std::vector<std::string> _required_label_names;

  }; // class Builder


private:
  enum DORadType {enum_linearSolve, enum_sweepSpatiallyParallel};
  int _nDir;
  int _nphase;
  int _nstage;
  int _nsteps_calc_freq;                // number of radiation solves to resolve the radiation time-scale

  bool _multiBox;
  bool _dynamicSolveFrequency{false};  // turns on the radiation profiler, tool to identifying if radiation is being resolved

  std::vector<std::vector<double> > _xyzPatch_boundary;     // all patch boundaries (approximate), needed for multi-box weeps,

  std::vector< std::vector < std::vector < bool > > > _doesPatchExist;
  std::vector<const PatchSubset*> _RelevantPatchesXpYpZp;   // Some redundancy here, since XpYpZp = XmYmZm [ end : start ]
  std::vector<const PatchSubset*> _RelevantPatchesXpYpZm;   // only need four sets...
  std::vector<const PatchSubset*> _RelevantPatchesXpYmZp;
  std::vector<const PatchSubset*> _RelevantPatchesXpYmZm;
  std::vector<const PatchSubset*> _RelevantPatchesXmYpZp;
  std::vector<const PatchSubset*> _RelevantPatchesXmYpZm;
  std::vector<const PatchSubset*> _RelevantPatchesXmYmZp;
  std::vector<const PatchSubset*> _RelevantPatchesXmYmZm;

  std::vector<const PatchSet*> _RelevantPatchesXpYpZp2;     // Some redundancy here, since XpYpZp = XmYmZm [ end : start ]
  std::vector<const PatchSet*> _RelevantPatchesXpYpZm2;     // only need four sets...
  std::vector<const PatchSet*> _RelevantPatchesXpYmZp2;
  std::vector<const PatchSet*> _RelevantPatchesXpYmZm2;
  std::vector<const PatchSet*> _RelevantPatchesXmYpZp2;
  std::vector<const PatchSet*> _RelevantPatchesXmYpZm2;
  std::vector<const PatchSet*> _RelevantPatchesXmYmZp2;
  std::vector<const PatchSet*> _RelevantPatchesXmYmZm2;


  IntVector _patchIntVector;
  int _radiation_calc_freq;
  int _nQn_part;

  bool _all_rk;
  bool _using_prop_calculator;
  bool _checkForMissingIntensities;
  int _sweepMethod;
  std::vector <std::vector< std::vector<int> > > _directional_phase_adjustment;

  std::string _profiler_label_name;
  std::string _T_label_name;
  std::string _abskt_label_name;
  std::string _abskg_label_name;
  //std::vector<std::string> _abskg_label_name{1};
  int d_nbands{1};

  DORadiationModel* _DO_model;
  ArchesLabel*    _labels;
  MPMArchesLabel* _MAlab;
  RadPropertyCalculator* _prop_calculator;
  const ProcessorGroup*  _my_world;

  const MaterialSet* m_matls{nullptr};                    ///< Arches material set
  int m_matIdx{-9};                                       ///< Arches material index

  bool m_user_intensity_save{false};                      ///< True if the user requests to save specific intensities
                                                          //   This changes the behavior of the carry forward in the case
                                                          //   when scattering isn't on.
                                                          
  std::vector<const VarLabel*> m_user_intensity_save_labels; ///< Labels the user has requested to be saved.
  std::vector<const VarLabel*> _missingCkPt_labels;       // labels that are missing from the checkpoint


  std::vector<const VarLabel*> _species_varlabels;
  std::vector<const VarLabel*> _size_varlabels;
  std::vector<const VarLabel*> _w_varlabels;
  std::vector<const VarLabel*> _T_varlabels;

  const VarLabel* _scatktLabel;
  const VarLabel* _asymmetryLabel;
  const VarLabel* _T_label;
  const VarLabel* _abskt_label;
  const VarLabel* _abskg_label;
  const VarLabel* _radiationSRCLabel;
  const VarLabel* _radiationFluxELabel;
  const VarLabel* _radiationFluxWLabel;
  const VarLabel* _radiationFluxNLabel;
  const VarLabel* _radiationFluxSLabel;
  const VarLabel* _radiationFluxTLabel;
  const VarLabel* _radiationFluxBLabel;
  const VarLabel* _radiationVolqLabel;
  const PatchSet* _perproc_patches;
  std::vector< std::vector < std::vector < Ghost::GhostType > > >   _gv;

  const std::string dynamicSolveCountPatch_name {"dynamicSolveCountPatch"};
  const VarLabel* _dynamicSolveCountPatchLabel;
  const VarLabel* _lastRadSolvePatchLabel;

  std::vector<const VarLabel*>  _radIntSource;
  std::vector<std::string> _radIntSource_names;

  std::vector< const VarLabel*> _IntensityLabels;
  std::vector< const VarLabel*> _emiss_plus_scat_source_label;

  std::vector< std::vector< const VarLabel*> > _patchIntensityLabels;

  ApplicationInterface* m_arches{nullptr};
  //__________________________________
  //
  int intensityIndx(const int dir,
                    const int iband);


}; // end DORadiation
} // end namespace Uintah
#endif
