#ifndef Uintah_Component_Arches_RMCRT_h
#define Uintah_Component_Arches_RMCRT_h
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/Radiation/RadPropertyCalculator.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/MPMArches/MPMArchesLabel.h>

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
* TO DO'S: 
*  @todo Remove _bc from code.  But first must remove it from RMCRT_RadiationModel.cc
*  
*/ 

namespace Uintah{
 
  class ArchesLabel; 
  class BoundaryCondition; 

class RMCRT_Radiation: public SourceTermBase {
public: 

  RMCRT_Radiation( std::string srcName, 
                   ArchesLabel* labels, 
                   MPMArchesLabel* MAlab, 
                   BoundaryCondition* bc, 
                   vector<std::string> reqLabelNames, 
                   const ProcessorGroup* my_world, 
                   std::string type );
                   
  ~RMCRT_Radiation();

  void problemSetup(const ProblemSpecP& db);
  
  void extraSetup();
  
  void sched_computeSource( const LevelP& level, 
                            SchedulerP& sched, 
                            int timeSubStep );
                            
  void computeSource( const ProcessorGroup*, 
                      const PatchSubset* , 
                      const MaterialSubset* , 
                      DataWarehouse* , 
                      DataWarehouse* , 
                      int timeSubStep );
  
  void sched_dummyInit( const LevelP& level, 
                        SchedulerP& sched );
  
  void dummyInit( const ProcessorGroup* pc, 
                  const PatchSubset* patches, 
                  const MaterialSubset* matls,
                  DataWarehouse* old_dw, 
                  DataWarehouse* new_dw );

  //______________________________________________________________________
  class Builder
    : public SourceTermBase::Builder { 

    public: 

      Builder( std::string name, 
               vector<std::string> required_label_names, 
               ArchesLabel* labels, 
               BoundaryCondition* bc, 
               const ProcessorGroup* my_world ) 
        : _name(name), 
          _labels(labels), 
          _bc(bc), 
          _my_world(my_world), 
          _required_label_names(required_label_names){
          _type = "rmcrt_radiation"; 
        };
      ~Builder(){}; 

      RMCRT_Radiation* build()
      { 
        return scinew RMCRT_Radiation( _name, _labels, _MAlab, _bc, _required_label_names, _my_world, _type ); 
      };

    private: 
      std::string         _name; 
      std::string         _type; 
      ArchesLabel*        _labels; 
      MPMArchesLabel*     _MAlab;
      BoundaryCondition*  _bc; 
      const ProcessorGroup* _my_world; 
      vector<std::string> _required_label_names;

  }; // class Builder 
  //______________________________________________________________________
  //
private:


  void sched_radProperties( const LevelP& level,
                            SchedulerP& sched,
                            const int time_sub_step );
  
  void radProperties( const ProcessorGroup* ,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* ,
                      DataWarehouse* new_dw,
                      const int time_sub_step );    


  //__________________________________
  //
  int _radiation_calc_freq;
  int _matl;
  MaterialSet* _matlSet;
  

  bool _all_rk; 
  bool _using_prop_calculator;
  
  int  _whichAlgo;
  enum Algorithm{ dataOnion, coarseLevel};
  
  double _initAbskg; 

  Ray* _RMCRT;
  ArchesLabel*    _labels; 
  MPMArchesLabel* _MAlab;
  BoundaryCondition* _bc; 
  RadPropertyCalculator* _prop_calculator; 
  const ProcessorGroup* _my_world;
  SimulationStateP      _sharedState;
  ProblemSpecP          _ps;  // needed for extraSetup()
 
  const VarLabel* _T_label;
  const VarLabel* _tempLabel;
  const VarLabel* _sigmaT4Label;
  const VarLabel* _abskgLabel;
  const VarLabel* _absorpLabel;
  const VarLabel* _cellTypeLabel; 
  
  Ghost::GhostType _gn;
  Ghost::GhostType _gac;

	std::vector<const VarLabel*> _species_varlabels; 

}; // end RMCRT
} // end namespace Uintah
#endif
