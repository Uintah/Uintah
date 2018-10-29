/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

  void sched_restartInitialize( const LevelP& level, SchedulerP& sched );

  void initialize( const ProcessorGroup* pg,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw );

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
        : _name(name),
          _labels(labels),
          _my_world(my_world),
          _required_label_names(required_label_names)
        {}

      ~Builder(){}

      RMCRT_Radiation* build()
      {
        return scinew RMCRT_Radiation( _name, _labels, _MAlab, _required_label_names, _my_world, _type );
      }

    private:

      std::string         _name;
      std::string         _type{"rmcrt_radiation"};
      ArchesLabel*        _labels{nullptr};
      MPMArchesLabel*     _MAlab{nullptr};
      const ProcessorGroup* _my_world;
      std::vector<std::string> _required_label_names;
  }; // class Builder

  //______________________________________________________________________
  //
private:

  //
  void restartInitializeHack( const ProcessorGroup*, 
                              const PatchSubset*,
                              const MaterialSubset*, 
                              DataWarehouse*, 
                              DataWarehouse*);



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
                  coarseLevel, 
                  singleLevel};

  int  _matl;
  int  _archesLevelIndex{-9};
  bool _all_rk{false};

  int  _whichAlgo{singleLevel};

  Ray                  * _RMCRT{nullptr};
  ArchesLabel          * _labels{nullptr};
  MPMArchesLabel       * _MAlab{nullptr};
  BoundaryCondition    * _boundaryCondition{nullptr};
  Properties           * d_props{nullptr};
  const ProcessorGroup * _my_world;
  MaterialManagerP       _materialManager;
  ProblemSpecP           _ps;              // needed for extraSetup()

  std::string  _abskt_label_name;
  std::string  _T_label_name;

  const VarLabel * _abskgLabel{nullptr};
  const VarLabel * _absktLabel{nullptr};
  const VarLabel * _tempLabel{nullptr};
  const VarLabel * _radFluxE_Label{nullptr};
  const VarLabel * _radFluxW_Label{nullptr};
  const VarLabel * _radFluxN_Label{nullptr};
  const VarLabel * _radFluxS_Label{nullptr};
  const VarLabel * _radFluxT_Label{nullptr};
  const VarLabel * _radFluxB_Label{nullptr};

#if 0
  // variables needed for particles
  bool _radiateAtGasTemp{true};  // this flag is arbitrary for no particles

  std::vector<std::string>       _temperature_name_vector;
  std::vector<std::string>       _absk_name_vector;
  std::vector< const VarLabel*>  _absk_label_vector;
  std::vector< const VarLabel*>  _temperature_label_vector;

  int _nQn_part{0} ;  // number of quadrature nodes in DQMOM
#endif
  Ghost::GhostType _gn{Ghost::None};
  Ghost::GhostType _gac{Ghost::AroundCells};
  
  TypeDescription::Type _FLT_DBL{TypeDescription::double_type};        // Is RMCRT algorithm using doubles or floats for communicated variables

}; // end RMCRT

} // end namespace Uintah

#endif // CCA_COMPONENTS_ARCHES_SOURCETERMS_RMCRT_H
