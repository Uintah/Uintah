#ifndef Uintah_Component_Arches_ModelBase_h
#define Uintah_Component_Arches_ModelBase_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/ArchesMaterial.h>


//===============================================================

/** 
  * @class  ModelBase
  * @author Jeremy Thornock, Charles Reid
  * @date   November 2008, November 2009
  * 
  * @brief A base class for models for a transport 
  *        equation. 
  * 
  */ 

namespace Uintah {

class ArchesLabel;

class ModelBase{ 

public: 

  ModelBase( std::string modelName, 
             SimulationStateP& sharedState, 
             ArchesLabel* fieldLabels,
             vector<std::string> icLabelNames, 
             vector<std::string> scalarLabelNames, 
             int qn );
  
  virtual ~ModelBase();

  ///////////////////////////////////////////////
  // Initialization methods

  /** @brief  Input file interface */
  virtual void problemSetup(const ProblemSpecP& db, int qn) = 0;  

  /** @brief  Pure virtual function: schedule initialization of any special variables unique to the model. */ 
  virtual void sched_initVars( const LevelP&  level, 
                               SchedulerP&    sched ) = 0;

  /** @brief  Pure virtual fucntion: actually initialize any special variables unique to the model. */
  virtual void initVars( const ProcessorGroup * pc,
                         const PatchSubset    * patches, 
                         const MaterialSubset * matls, 
                         DataWarehouse        * old_dw, 
                         DataWarehouse        * new_dw ) = 0;


  /** @brief  Schedule dummy initialization for MPMARCHES; the schedule task is the same for all models,
              but the implementation must be done by each model, since knowledge of the model's data type is required.
      @see    ExplicitSolver::noSolve() */
  virtual void sched_dummyInit( const LevelP& level, SchedulerP& sched ) = 0;

  /** @breif  Pure virtual function: actually do the dummy initialization */
  virtual void dummyInit( const ProcessorGroup * pc, 
                          const PatchSubset    * patches, 
                          const MaterialSubset * matls, 
                          DataWarehouse        * old_dw, 
                          DataWarehouse        * new_dw ) = 0;

  /** @brief  Reinitialize the flags that tells the scheduler if the varLabel needs a compute or a modifies. */
  // Note I need two of these flags; 1 for scheduling and 1 for actual execution.
  inline void reinitializeLabel(){ 
    d_labelSchedInit  = false; };

  ////////////////////////////////////////////////
  // Model computation methods

  /** @brief  Pure virtual function: schedule computation of DQMOM model term. */
  virtual void sched_computeModel(const LevelP& level, 
                                  SchedulerP&   sched, 
                                  int           timeSubStep ) = 0;

  /** @brief  Pure virtual function: actually compute the DQMOM model term. */
  virtual void computeModel( const ProcessorGroup * pc,
                             const PatchSubset    * patches,
                             const MaterialSubset * matls, 
                             DataWarehouse        * old_dw, 
                             DataWarehouse        * new_dw ) = 0;

  ///////////////////////////////////////////////////
  // Access methods

  /** @brief  Return a string containing the model type (pure virtual) */
  virtual string getType() = 0;

  /** @brief  Return the VarLabel for the model term for particle */
  inline const VarLabel* getModelLabel() {
    return d_modelLabel; };

  /** @brief  Return the VarLabel for the model term for gas */
  inline const VarLabel* getGasSourceLabel() {
    return d_gasLabel; }; 

  /** @brief  Return the quadrature node */
  inline int getquadNode() {
    return d_quadNode; };
 
  inline void setUnweightedAbscissas(bool d_unw){
    d_unweighted = d_unw;
  };

protected:

  std::string d_modelName; 
  
  SimulationStateP& d_sharedState; 

  ArchesLabel* d_fieldLabels;

  vector<string> d_icLabels;          ///< All required internal coordinate labels (from DQMOM factory) needed to compute this model
  vector<string> d_scalarLabels;      ///< All required scalar labels (from scalarFactory) needed to compute this model
  vector<const VarLabel*> _extra_local_labels; ///< All new local labels that the model uses/needs/computes
  map<string, string> LabelToRoleMap; ///< Map of internal coordinate or scalar labels to their role in the model

  const VarLabel* d_modelLabel;       ///< Label storing the value of this model
  const VarLabel* d_gasLabel;         ///< Label for gas phase source term 
  int d_timeSubStep;

  double d_lowModelClip;              ///< All models should have capability of clipping low values
  double d_highModelClip;             ///< All models should have capability of clipping high values

  bool d_labelSchedInit;
  bool d_labelActualInit;   
  bool d_unweighted;

  int d_quadNode; 

}; // end ModelBase
}  // end namespace Uintah

#endif
