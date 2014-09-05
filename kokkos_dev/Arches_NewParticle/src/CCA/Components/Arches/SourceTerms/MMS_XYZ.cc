#include <CCA/Components/Arches/SourceTerms/MMS_XYZ.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Grid/Level.h>

//===========================================================================

using namespace std;
using namespace Uintah; 

/** @details
The MMS_XYZ class creates a source term Q for a manufactured solution testing only
the X convection term.  It is a source term for the PDE:

\f[
u \frac{ \partial \phi }{ \partial x } + v \frac{ \partial \phi }{ \partial y } + w \frac{ \partial \phi }{ \partial z }
= Q
\f]

Note that this PDE is steady, incompressible, and assumes constant velocity.

The unsteady term is zero because the assumed solution is not a function of time.

These conditions (steady, incompressible, constant X velocity) are achieved 
by using a constant velocity and density field using the constant MMS case, 
whose input file is located in:

StandAlone/inputs/ARCHES/mms/constantMMS.ups

The assumed solution is:

\f[
\phi = \sin ( 2 \pi \frac{x}{L_x} ) \cos ( 2 \pi \frac{y}{L_y} ) \sin ( 2 \pi \frac{z}{L_z} )
\f]

where \f$L_i\f$ is the length of the domain in the \f$i^{th}\f$ direction.

This makes the source term Q equal to:

\f[
Q = 2 \pi u \frac{1}{L_x} \cos ( 2 \pi \frac{x}{L_x} ) \cos ( 2 \pi \frac{y}{L_y} ) \sin ( 2 \pi \frac{z}{L_z} ) 
  - 2 \pi v \frac{1}{L_y} \sin ( 2 \pi \frac{x}{L_x} ) \sin ( 2 \pi \frac{y}{L_y} ) \sin ( 2 \pi \frac{z}{L_z} )
  + 2 \pi w \frac{1}{L_z} \sin ( 2 \pi \frac{x}{L_x} ) \cos ( 2 \pi \frac{y}{L_y} ) \cos ( 2 \pi \frac{z}{L_z} ) 
\f]

*/
MMS_XYZ::MMS_XYZ( std::string srcName, SimulationStateP& sharedState,
                            vector<std::string> reqLabelNames ) 
: SourceTermBase(srcName, sharedState, reqLabelNames)
{
  _src_label = VarLabel::create(srcName, CCVariable<double>::getTypeDescription()); 

  _source_type = CC_SRC; 

  MMS_XYZ::pi = 3.1415926535;
}

MMS_XYZ::~MMS_XYZ()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
MMS_XYZ::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb;
  ProblemSpecP db_root = db->getRootNode();

  if( db_root->findBlock("CFD")->findBlock("ARCHES") ) {
    if( db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("MMS") ) {
      ProblemSpecP db_mms = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("MMS");

      string which_mms;
      if( !db_mms->getAttribute( "whichMMS", which_mms ) ) {
        throw ProblemSetupException( "ERROR: Arches: MMS_XYZ: No Arches MMS type specified in input file.  To use MMS_XYZ, you must use a constant MMS.  See 'StandAlone/inputs/ARCHES/mms/constantMMS.ups' for an example input.", __FILE__, __LINE__);      
      } 

      if( which_mms != "constantMMS" ) {
        throw ProblemSetupException( "ERROR: Arches: MMS_XYZ: Incorrect Arches MMS type specified in input file.  To use MMS_XYZ, you must use a constant MMS.  See 'StandAlone/inputs/ARCHES/mms/constantMMS.ups' for an example input.", __FILE__, __LINE__); 
      }

      ProblemSpecP db_const_mms = db_mms->findBlock("constantMMS");
      db_const_mms->getWithDefault("cu",MMS_XYZ::d_uvel,0.0);
      db_const_mms->getWithDefault("cv",MMS_XYZ::d_vvel,0.0);
      db_const_mms->getWithDefault("cw",MMS_XYZ::d_wvel,0.0);

    } //end findblock MMS
  } //end findblock Arches
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
MMS_XYZ::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "MMS_XYZ::computeSource";
  Task* tsk = scinew Task(taskname, this, &MMS_XYZ::computeSource, timeSubStep);

  if (timeSubStep == 0 && !_label_sched_init) {
    // Every source term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    _label_sched_init = true;
  }

  if( timeSubStep == 0 ) {
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }

  grid = level->getGrid();

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
MMS_XYZ::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  Vector domain_size = Vector(0.0,0.0,0.0);
  grid->getLength(domain_size);

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 
    Vector Dx = patch->dCell(); 

    CCVariable<double> mmsXYZSrc; 

    if( timeSubStep == 0 ) {
      new_dw->allocateAndPut( mmsXYZSrc, _src_label, matlIndex, patch );
      mmsXYZSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( mmsXYZSrc, _src_label, matlIndex, patch ); 
    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      double x = c[0]*Dx.x() + Dx.x()/2.0; 
      double y = c[1]*Dx.y() + Dx.y()/2.0;
      double z = c[2]*Dx.z() + Dx.z()/2.0;

      mmsXYZSrc[c] = MMS_XYZ::evaluate_MMS_source( x, y, z, domain_size );
    }
  }
}
//---------------------------------------------------------------------------
// Method: Schedule dummy initialization
//---------------------------------------------------------------------------
void
MMS_XYZ::sched_dummyInit( const LevelP& level, SchedulerP& sched )
{
  string taskname = "MMS_XYZ::dummyInit"; 

  Task* tsk = scinew Task(taskname, this, &MMS_XYZ::dummyInit);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
MMS_XYZ::dummyInit( const ProcessorGroup* pc, 
                      const PatchSubset* patches, 
                      const MaterialSubset* matls, 
                      DataWarehouse* old_dw, 
                      DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;

    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, *iter, matlIndex, patch ); 
    }
  }
}

double MMS_XYZ::d_uvel; 
double MMS_XYZ::d_vvel; 
double MMS_XYZ::d_wvel; 
double MMS_XYZ::pi;

