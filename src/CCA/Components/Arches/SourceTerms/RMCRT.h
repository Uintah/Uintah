/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef CCA_COMPONENTS_ARCHES_SOURCETERMS_RMCRT_H
#define CCA_COMPONENTS_ARCHES_SOURCETERMS_RMCRT_H

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Properties.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Models/Radiation/RMCRT/Ray.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>

#include <Core/Grid/MaterialManagerP.h>
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
*    <nDivQRays>         25         </nDivQRays>
*    <Threshold>         0.05       </Threshold>
*    <rayDirSampleAlgo>  Naive      </rayDirSampleAlgo>
*    <StefanBoltzmann>   5.67051e-8 </StefanBoltzmann>
* </RMCRT>
*
*
*/


//______________________________________________________________________
//
namespace Uintah {

  class ArchesLabel;
  class BoundaryCondition;

class RMCRT_Radiation: public SourceTermBase {
public:

  using string_vector = std::vector<std::string>;

  RMCRT_Radiation( std::string      srcName,
                   ArchesLabel    * labels,
                   MPMArchesLabel * MAlab,
                   string_vector    reqLabelNames,
                   const ProcessorGroup * my_world,
                   std::string      type);


  ~RMCRT_Radiation();

  void problemSetup( const ProblemSpecP& db );

  void extraSetup( GridP& grid, BoundaryCondition* bc, TableLookup* prop );

  void sched_computeSource( const LevelP& level,
                            SchedulerP& sched,
                            int timeSubStep );

  void computeSource( const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      int timeSubStep );

  void sched_initialize( const LevelP& level, SchedulerP& sched );

  void initialize( const ProcessorGroup* pg,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

  void sched_restartInitialize( const LevelP& level, SchedulerP& sched );





  enum GRAPH_TYPE {
      TG_CARRY_FORWARD = 0              // carry forward taskgraph
    , TG_RMCRT         = 1              // RMCRT radiation taskgraph
    , NUM_GRAPHS
  };

  //______________________________________________________________________
  class Builder : public SourceTermBase::Builder {

    public:

      Builder( std::string name,
               std::vector<std::string> required_label_names,
               ArchesLabel* labels,
               const ProcessorGroup* my_world )
        : m_name(name),
          m_labels(labels),
          m_my_world(my_world),
          m_required_label_names(required_label_names)
        {}

      ~Builder(){}

      RMCRT_Radiation* build()
      {
        return scinew RMCRT_Radiation( m_name, m_labels, m_MAlab, m_required_label_names, m_my_world, m_type );
      }

    private:

      std::string         m_name;
      std::string         m_type{"rmcrt_radiation"};
      ArchesLabel*        m_labels{nullptr};
      MPMArchesLabel*     m_MAlab{nullptr};
      const ProcessorGroup* m_my_world;
      std::vector<std::string> m_required_label_names;
  }; // class Builder

  //______________________________________________________________________
  //
private:

  // These tasks are used to "fake" out the taskgraph createDetailedDependency() logic
  // On a restart, before you can require something from the new_dw there must be a compute() for that
  // variable.
  void restartInitializeHack( const ProcessorGroup*,
                              const PatchSubset*,
                              const MaterialSubset*,
                              DataWarehouse*,
                              DataWarehouse*){};

  void restartInitializeHack2( const ProcessorGroup*,
                               const PatchSubset*,
                               const MaterialSubset*,
                               DataWarehouse*,
                               DataWarehouse*){};

  //__________________________________
  //
  void restartInitialize( const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw );

  //__________________________________
  /** @brief Schedule compute of blackbody intensity */
  void sched_sigmaT4( const LevelP  & level,
                      SchedulerP    & sched );

  //__________________________________
  //
  template< class T>
  void sigmaT4( const ProcessorGroup * pg,
                const PatchSubset    * patches,
                const MaterialSubset * matls,
                DataWarehouse        * old_dw,
                DataWarehouse        * new_dw,
                Task::WhichDW          which_dw );

  //__________________________________
  /** @brief Schedule compute of blackbody intensity */
  void sched_sumAbsk( const LevelP  & level,
                      SchedulerP    & sched  );

  //__________________________________
  //
  template< class T>
  void sumAbsk( const ProcessorGroup * pg,
                const PatchSubset    * patches,
                const MaterialSubset * matls,
                DataWarehouse        * old_dw,
                DataWarehouse        * new_dw,
                Task::WhichDW          which_dw );

  //______________________________________________________________________
  //   Boundary Conditions
  void  sched_setBoundaryConditions( const LevelP& level,
                                     SchedulerP& sched,
                                     Task::WhichDW temp_dw,
                                     const bool backoutTemp = false);

    template< class T >
    void setBoundaryConditions( const ProcessorGroup* pg,
                                const PatchSubset* patches,
                                const MaterialSubset*,
                                DataWarehouse*,
                                DataWarehouse* new_dw,
                                Task::WhichDW temp_dw,
                                const bool backoutTemp );

     //__________________________________
     //  move CCVariable<stencil7> -> 6 CCVariable<double>
    void sched_stencilToDBLs( const LevelP& level,
                              SchedulerP& sched );

    void stencilToDBLs( const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw );

    void sched_fluxInit( const LevelP& level,
                         SchedulerP& sched );

    void fluxInit( const ProcessorGroup* pg,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

    //__________________________________
    //  move  6 CCVariable<double> -> CCVariable<stencil7>
    void sched_DBLsToStencil( const LevelP& level,
                              SchedulerP& sched );

    void DBLsToStencil( const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw );

  //__________________________________
  //

  enum Algorithm{ dataOnion,
                  dataOnionSlim,
                  coarseLevel,
                  singleLevel,
                  radiometerOnly       // VRFlux is computed at radiometer locations
                };
  int  m_matl;
  int  m_archesLevelIndex{-9};
  bool m_all_rk{false};

  int  m_whichAlgo{singleLevel};

  Ray                  * m_RMCRT{nullptr};
  ArchesLabel          * m_labels{nullptr};
  MPMArchesLabel       * m_MAlab{nullptr};
  BoundaryCondition    * m_boundaryCondition{nullptr};
  Properties           * d_props{nullptr};
  const ProcessorGroup * m_my_world;
  MaterialManagerP       m_materialManager;
  ProblemSpecP           m_ps;                   // needed for extraSetup()
  const MaterialSet    * m_matlSet{nullptr};       //< Arches material set

  const VarLabel * m_gasTemp_Label{nullptr};
  const VarLabel * m_sumAbsk_Label{nullptr};

  const VarLabel * m_radFluxE_Label{nullptr};
  const VarLabel * m_radFluxW_Label{nullptr};
  const VarLabel * m_radFluxN_Label{nullptr};
  const VarLabel * m_radFluxS_Label{nullptr};
  const VarLabel * m_radFluxT_Label{nullptr};
  const VarLabel * m_radFluxB_Label{nullptr};

  // variables needed for radiation from particles
  bool m_radiateAtGasTemp{true};  // this flag is arbitrary for no particles
  bool m_do_partRadiation{false};
  std::vector<std::string>       m_partGas_temp_names;
  std::vector<std::string>       m_partGas_absk_names;
  std::vector< const VarLabel*>  m_partGas_absk_Labels;
  std::vector< const VarLabel*>  m_partGas_temp_Labels;
  std::map< const VarLabel*, double>  m_missingCkPt_Labels;   // map that contains varLabel and initialization value

  int m_nQn_part{0};
  int m_nPartGasLabels{1};                                // Always at least 1 label

  Ghost::GhostType m_gn{Ghost::None};
  Ghost::GhostType m_gac{Ghost::AroundCells};

  TypeDescription::Type m_FLT_DBL{TypeDescription::double_type};        // Is RMCRT algorithm using doubles or floats for communicated variables

}; // end RMCRT

} // end namespace Uintah

#endif // CCA_COMPONENTS_ARCHES_SOURCETERMS_RMCRT_H
