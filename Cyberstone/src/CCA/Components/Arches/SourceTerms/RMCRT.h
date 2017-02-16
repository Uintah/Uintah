#ifndef Uintah_Component_Arches_RMCRT_h
#define Uintah_Component_Arches_RMCRT_h
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/ProblemSpec/ProblemSpec.h>


/**
* @class  RMCRT_Radiation
* @author Todd Harman
* @date   March 2012
*
* @brief Computes the divergence of heat flux contribution from the
*         solution of the intensity equation.
*
* The input file interface for this property should like this in your UPS file:
*
*  <calc_frequency />
*  <calc_on_all_RKsteps/> <!-- calculate radiation every RK step, default = false -->
*
*
*  <RMCRT>
*    <randomSeed>        false      </randomSeed>
*    <NoOfRays>          25         </NoOfRays>
*    <Threshold>         0.05       </Threshold>
*    <Slice>             20         </Slice>
*    <StefanBoltzmann>   5.67051e-8 </StefanBoltzmann>
* </RMCRT>
*
*
*/
//______________________________________________________________________
//
namespace Uintah{

  class ArchesLabel;
  class BoundaryCondition;

class RMCRT_Radiation: public SourceTermBase {
public:

  RMCRT_Radiation( std::string srcName,
                   ArchesLabel* labels,
                   MPMArchesLabel* MAlab,
                   std::vector<std::string> reqLabelNames,
                   const ProcessorGroup* my_world,
                   std::string type );

  ~RMCRT_Radiation();

  void problemSetup(const ProblemSpecP& db );

  void extraSetup( GridP& grid, BoundaryCondition* bc, TableLookup* prop );

  void sched_computeSource( const LevelP& level,
                            SchedulerP& sched,
                            int timeSubStep );

  void computeSource( const ProcessorGroup*,
                      const PatchSubset* ,
                      const MaterialSubset* ,
                      DataWarehouse* ,
                      DataWarehouse* ,
                      int timeSubStep );

  void sched_initialize( const LevelP& level,
                         SchedulerP& sched );

  void sched_restartInitialize( const LevelP& level,
                                SchedulerP& sched );

  void initialize( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  //______________________________________________________________________
  class Builder
    : public SourceTermBase::Builder {

    public:

      Builder( std::string name,
               std::vector<std::string> required_label_names,
               ArchesLabel* labels,
               const ProcessorGroup* my_world )
        : _name(name),
          _labels(labels),
          _my_world(my_world),
          _required_label_names(required_label_names){
          _type = "rmcrt_radiation";
        }

      ~Builder(){}

      RMCRT_Radiation* build()
      {
        return scinew RMCRT_Radiation( _name, _labels, _MAlab, _required_label_names, _my_world, _type );
      }

    private:

      std::string         _name;
      std::string         _type;
      ArchesLabel*        _labels;
      MPMArchesLabel*     _MAlab;
      const ProcessorGroup* _my_world;
      std::vector<std::string> _required_label_names;
  }; // class Builder
  //______________________________________________________________________
  //
private:

    //
    void restartInitializeHack( const ProcessorGroup* , const PatchSubset* ,
                                   const MaterialSubset* , DataWarehouse*, DataWarehouse*);



     //______________________________________________________________________
     //   Boundary Conditions
    void  sched_setBoundaryConditions( const LevelP& level,
                                       SchedulerP& sched,
                                       Task::WhichDW temp_dw,
                                       const int radCalc_freq,
                                       const bool backoutTemp = false);

    template< class T >
    void setBoundaryConditions( const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw,
                                Task::WhichDW temp_dw,
                                const int radCalc_freq,
                                const bool backoutTemp );

     //__________________________________
     //  move CCVariable<stencil7> -> 6 CCVariable<double>
    void sched_stencilToDBLs( const LevelP& level,
                              SchedulerP& sched );

    void stencilToDBLs( const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw );

    void sched_fluxInit( const LevelP& level,
                         SchedulerP& sched );

    void fluxInit( const ProcessorGroup* pc,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

    //__________________________________
    //  move  6 CCVariable<double> -> CCVariable<stencil7>
    void sched_DBLsToStencil( const LevelP& level,
                              SchedulerP& sched );

    void DBLsToStencil( const ProcessorGroup* pc,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw );

  //__________________________________
  //
  int _radiation_calc_freq;
  int _matl;
  int _archesLevelIndex;
  bool _all_rk;

  int  _whichAlgo;
  enum Algorithm{ dataOnion, coarseLevel, singleLevel};

  Ray* _RMCRT;
  ArchesLabel*    _labels;
  MPMArchesLabel* _MAlab;
  BoundaryCondition* _boundaryCondition;
  Properties* d_props;
  const ProcessorGroup* _my_world;
  SimulationStateP      _sharedState;
  ProblemSpecP          _ps;              // needed for extraSetup()

  std::string _abskg_label_name;
  std::string _T_label_name;

  const VarLabel* _abskgLabel;
  const VarLabel* _tempLabel;
  const VarLabel* _radFluxE_Label;
  const VarLabel* _radFluxW_Label;
  const VarLabel* _radFluxN_Label;
  const VarLabel* _radFluxS_Label;
  const VarLabel* _radFluxT_Label;
  const VarLabel* _radFluxB_Label;

  Ghost::GhostType _gn;
  Ghost::GhostType _gac;
  TypeDescription::Type _FLT_DBL;        // Is RMCRT algorithm using doubles or floats for communicated variables

}; // end RMCRT
} // end namespace Uintah
#endif
