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

#include <CCA/Components/OnTheFlyAnalysis/meanTurbFluxes.h>
#include <CCA/Components/OnTheFlyAnalysis/FileInfoVar.h>

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/OS/Dir.h> // for MKDIR
#include <Core/Util/FileUtils.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdio>

#define ALL_LEVELS 99
#define FINEST_LEVEL -1
using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "meanTurbFluxes:+"
Dout dbg_OTF_MTF("meanTurbFluxes", "OnTheFlyAnalysis", "meanTurbFluxes debug stream", false);

//______________________________________________________________________
/*

______________________________________________________________________*/

meanTurbFluxes::meanTurbFluxes( const ProcessorGroup    * myworld,
                                const MaterialManagerP    materialManager,
                                const ProblemSpecP      & module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set  = nullptr;

  d_monitorCell = IntVector(0,0,0);

  d_planeAve_1 = scinew planeAverage( myworld, materialManager, module_spec, true,  false, 0);
  d_planeAve_2 = scinew planeAverage( myworld, materialManager, module_spec, false, true,  1);

  d_lb        = scinew meanTurbFluxesLabel();

  d_lb->lastCompTimeLabel =  VarLabel::create("lastCompTime_planeAve",
                                              max_vartype::getTypeDescription() );

  d_velVar =std::make_shared< velocityVar >();
}

//__________________________________
meanTurbFluxes::~meanTurbFluxes()
{
  DOUT(dbg_OTF_MTF, " Doing: destorying meanTurbFluxes" );
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  VarLabel::destroy(d_lb->lastCompTimeLabel);

  delete d_lb;
  delete d_planeAve_1;
  delete d_planeAve_2;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void meanTurbFluxes::problemSetup(const ProblemSpecP &,
                                  const ProblemSpecP &,
                                  GridP & grid,
                                  std::vector<std::vector<const VarLabel* > > &PState,
                                  std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  DOUT(dbg_OTF_MTF, "Doing problemSetup \t\t\t\t meanTurbFluxes" );

  d_planeAve_1->setComponents( m_application );
  d_planeAve_2->setComponents( m_application );

  const ProblemSpecP & notUsed = {nullptr};
  d_planeAve_1->problemSetup( notUsed, notUsed, grid, PState, PState_preReloc);
  d_planeAve_2->problemSetup( notUsed, notUsed, grid, PState, PState_preReloc);

  //__________________________________
  //  Read in timing information
  m_module_spec->require("samplingFrequency", d_writeFreq);
  m_module_spec->require("timeStart",         d_startTime);
  m_module_spec->require("timeStop",          d_stopTime);
  // debugging
  m_module_spec->get(    "monitorCell",      d_monitorCell);

  ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("meanTurbFluxes: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  //__________________________________
  // find the material to extract data from.  Default is matl 0.
  // The user should specify
  //  <material>   atmosphere </material>
  const Material*  matl;
  if(m_module_spec->findBlock("material") ){
    matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  } else {
    matl = m_materialManager->getMaterial(0);
  }

  int defaultMatl = matl->getDWIndex();

  //__________________________________
  vector<int> m;
  m.push_back(0);            // matl for FileInfo label
  m.push_back(defaultMatl);

  // remove any duplicate entries
  sort(m.begin(), m.end());
  vector<int>::iterator it = unique(m.begin(), m.end());
  m.erase(it, m.end());

  //Construct the matl_set
  d_matl_set = scinew MaterialSet();
  d_matl_set->addAll(m);
  d_matl_set->addReference();



  //__________________________________
  //  velocity label
  map<string,string> attribute;
  ProblemSpecP vel_ps = m_module_spec->findBlock( "velocity" );

  if( vel_ps == nullptr ) {
    throw ProblemSetupException("meanTurbFluxes: velocity xml tag not found: ", __FILE__, __LINE__);
  }

  vel_ps->getAttributes( attribute );
  string labelName = attribute["label"];
  d_velVar->label   = VarLabel::find( labelName );

  if( d_velVar->label == nullptr ){
    throw ProblemSetupException("meanTurbFluxes: velocity label not found: " + labelName , __FILE__, __LINE__);
  }

  d_velVar->matl  = defaultMatl;
  d_velVar->level = ALL_LEVELS;
  
  const TypeDescription * td_V     = CCVariable<Vector>::getTypeDescription();
  d_velVar->primeLabel            = VarLabel::create( labelName + "_prime",         td_V);
  d_velVar->normalTurbStrssLabel  = VarLabel::create( d_velVar->normalTurbStrssName, td_V);
  d_velVar->shearTurbStrssLabel   = VarLabel::create( d_velVar->shearTurbStrssName,  td_V);





  typedef planeAverage PA;
  std::vector< std::shared_ptr< PA::planarVarBase > >planarVars;

  // normal turbulent stress
  PA::planarVar_Vector* pv = new PA::planarVar_Vector();
  pv->label      = d_velVar->normalTurbStrssLabel;
  pv->matl       = defaultMatl;
  pv->level      = ALL_LEVELS;
  pv->baseType   = td_V->getType();
  pv->subType    = TypeDescription::Vector;
  pv->weightType = PA::NCELLS;

  planarVars.push_back( std::shared_ptr< PA::planarVarBase >(pv) );

  // shear turbulent stress
//  pv->label      = d_velVar->shearTurbStrssLabel;
  
//  planarVars.push_back( std::shared_ptr< PA::planarVarBase >(pv) );







  //__________________________________
  //  Now loop over all the variables to be analyzed

  for( ProblemSpecP var_spec = vars_ps->findBlock( "analyze" ); var_spec != nullptr; var_spec = var_spec->findNextBlock( "analyze" ) ) {

    var_spec->getAttributes( attribute );

    //__________________________________
    // Read in the variable name
    string labelName = attribute["label"];
    VarLabel* label = VarLabel::find(labelName);
    if( label == nullptr ){
      throw ProblemSetupException("meanTurbFluxes: analyze label not found: " + labelName , __FILE__, __LINE__);
    }

    if (label == d_velVar->label ){  // velocity label has already been processed
      continue;
    }

    //__________________________________
    //  bulletproofing
    const TypeDescription* td = label->typeDescription();
    const TypeDescription* subtype = td->getSubType();

    const TypeDescription::Type baseType = td->getType();
    const TypeDescription::Type subType  = subtype->getType();

    // CC Variables, only Doubles and Vectors
    if(baseType != TypeDescription::CCVariable &&
       subType  != TypeDescription::double_type   ){
      ostringstream warn;
      warn << "ERROR:AnalysisModule:meanTurbFluxes: ("<<label->getName() << " "
           << " only CCVariable<double> variables work" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    //define intermediate quantity label names
    const TypeDescription * td_D   = CCVariable<double>::getTypeDescription();
    const TypeDescription * td_V   = CCVariable<Vector>::getTypeDescription();
    VarLabel* primeLabel     = VarLabel::create( labelName + "_prime",    td_D );        // Q'
    VarLabel* turbFluxLabel  = VarLabel::create( labelName + "_turbFlux", td_V );        // u'Q', v'Q', w'Q'

    // Bulletproofing - The user must specify the matl for single matl
    // variables
    if ( labelName == "press_CC" && attribute["matl"].empty() ){
      throw ProblemSetupException("meanTurbFluxes: You must add (matl='0') to the press_CC line." , __FILE__, __LINE__);
    }

    // Read in the optional level index
    int level = ALL_LEVELS;
    if (attribute["level"].empty() == false){
      level = atoi(attribute["level"].c_str());
    }

    //  Read in the optional material index
    int matl = defaultMatl;
    if (attribute["matl"].empty() == false){
      matl = atoi(attribute["matl"].c_str());
    }

    //__________________________________
    //  populate the vector of scalars
    Qvar* me = new Qvar();
    me->label         = label;
    me->primeLabel    = primeLabel;
    me->turbFluxLabel = turbFluxLabel;
    me->matl       = matl;
    me->level      = level;
    me->baseType   = baseType;
    me->subType    = subType;
    d_Qvars.push_back( std::shared_ptr<Qvar>(me) );
    
    
    PA::planarVar_Vector* pv = new PA::planarVar_Vector();
    pv->label      = turbFluxLabel;          // u'Q'(y), v'Q'(y), w'Q'(y)
    pv->matl       = matl;
    pv->level      = level;
    pv->baseType   = td_V->getType();
    pv->subType    = TypeDescription::Vector;
    pv->weightType = PA::NCELLS;

    planarVars.push_back( std::shared_ptr< PA::planarVarBase >(pv) );
  }
  
  d_planeAve_2->setAllLevels_planarVars( 0, planarVars );
}

//______________________________________________________________________
void meanTurbFluxes::scheduleInitialize(SchedulerP   & sched,
                                        const LevelP & level)
{
  d_planeAve_1->scheduleInitialize( sched, level);
  
  d_planeAve_2->scheduleInitialize( sched, level);
}



//______________________________________________________________________
void meanTurbFluxes::scheduleRestartInitialize(SchedulerP   & sched,
                                               const LevelP & level)
{
  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::scheduleRestartInitialize");

  d_planeAve_1->scheduleRestartInitialize( sched, level);
  
  d_planeAve_2->scheduleRestartInitialize( sched, level);
}

//______________________________________________________________________
void
meanTurbFluxes::restartInitialize()
{
}
//______________________________________________________________________
void meanTurbFluxes::scheduleDoAnalysis(SchedulerP   & sched,
                                        const LevelP & level)
{
  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::scheduleDoAnalysis");
  
  //__________________________________
  //  Not all ranks own patches, need custom MPI communicator
  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);
  
  //__________________________________
  // This instantiation of planarAve computes the planar averages of:
  // {u}^bar(y), {v}^bar(y), {w}^bar(y)
  // {Q}^bar(y)   Q = P, T, scalar.....etc  
  d_planeAve_1->createMPICommunicator( perProcPatches );
  
  d_planeAve_1->sched_computePlanarAve( sched, level );
  
  d_planeAve_1->sched_resetProgressVar( sched, level );
  
  d_planeAve_1->sched_updateTimeVar(    sched, level );


  //__________________________________
  //  compute u', v', w', Q'
  sched_TurbFluctuations( sched, level );
  
  
  //__________________________________
  //compute u'u', v'v', w'w'
  //        u'v', v'w', w'u'
  //        u'Q', v'Q', w'Q'
  sched_TurbFluxes(       sched, level );
  
  
  //__________________________________
  // This instantiation of planarAve computes:
  // {u'u'}^bar(y), {v'v'}^bar(y), {w'w'}^bar(y)      => normalTurbStrss
  // {u'v'}^bar(y), {v'w'}^bar(y), {w'u'}^bar(y)      => shearTurbStrss
  // {u'Q'}^bar(y), {v'Q'}^bar(y), {w'Q'}^bar(y)   
  d_planeAve_2->createMPICommunicator( perProcPatches );
  
  d_planeAve_2->sched_computePlanarAve( sched, level );
  
  d_planeAve_2->sched_resetProgressVar( sched, level );
  
  d_planeAve_2->sched_updateTimeVar(    sched, level );
  
}

//______________________________________________________________________
/*
    foreach y ( n_planes )
      iterate over all cells in Y plane {
        u' = u - u^bar(y)           Each plane in the grid will have a different _bar value
        v' = v - v^bar(y)           => CCVariable< Uintah:Vector >
        w' = w - w^bar(y)
      }
    end

    foreach y ( n_planes )
      foreach Q ( T, P, scalar )
        Q' = Q - Q^bar(y)          => CCVariable< double >
      end
    end
*/
//______________________________________________________________________
void meanTurbFluxes::sched_TurbFluctuations(SchedulerP   & sched,
                                            const LevelP & level)
{
  Task* t = scinew Task( "meanTurbFluxes::calc_TurbFluctuations",
                    this,&meanTurbFluxes::calc_TurbFluctuations );

  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::sched_TurbFluctuations");

  t->requires( Task::OldDW, m_simulationTimeLabel );
  t->requires( Task::OldDW, d_lb->lastCompTimeLabel );

  // u,v,w -> u',v',w'
  t->requires( Task::NewDW, d_velVar->label, d_velVar->matSubSet, Ghost::None, 0 );
  t->computes ( d_velVar->primeLabel );
  
  // Q -> Q'
  for ( unsigned int i =0 ; i < d_Qvars.size(); i++ ) {
    std::shared_ptr< Qvar > Q = d_Qvars[i];
    t->requires( Task::NewDW, Q->label, Q->matSubSet, Ghost::None, 0 );
    t->computes ( Q->primeLabel );
  }
  sched->addTask( t, level->eachPatch() , d_matl_set );
}

//______________________________________________________________________
//
void meanTurbFluxes::calc_TurbFluctuations(const ProcessorGroup  * ,
                                           const PatchSubset    * patches,
                                           const MaterialSubset * ,
                                           DataWarehouse        * ,
                                           DataWarehouse        * new_dw)
{
  for( auto p=0;p<patches->size();p++ ){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbg_OTF_MTF, "Doing meanTurbFluxes::calc_TurbFluctuations");

    // Q -> Q'
    for ( unsigned int i =0 ; i < d_Qvars.size(); i++ ) {
      std::shared_ptr< Qvar > Q = d_Qvars[i];
      calc_Q_prime< double >( new_dw, patch, Q );
    }

    // u,v,w -> u',v',w'
    calc_Q_prime< Vector >( new_dw, patch, d_velVar);
  }
}
//______________________________________________________________________
//
template <class T>
void meanTurbFluxes::calc_Q_prime( DataWarehouse         * new_dw,
                                   const Patch           * patch,
                                   std::shared_ptr<Qvar> Q)
{
  const int matl = Q->matl;

  constCCVariable<T> Qlocal;
  new_dw->get ( Qlocal, Q->label, matl, patch, Ghost::None, 0 );

  CCVariable< T > Qprime;
  new_dw->allocateAndPut( Qprime, Q->primeLabel, matl, patch );

  const Level* level = patch->getLevel();
  const int L_indx   = level->getIndex();
  std::vector< T > Qbar;
  
  d_planeAve_1->getPlanarAve< T >( L_indx, Q->label, Qbar );
  
  IntVector lo;
  IntVector hi;
  GridIterator iter=patch->getCellIterator();
  d_planeAve_1->planeIterator( iter, lo, hi );

  for ( auto z = lo.z(); z<hi.z(); z++ ) {          // This is the loop over all planes for this patch
    for ( auto y = lo.y(); y<hi.y(); y++ ) {        // cells in the plane
      for ( auto x = lo.x(); x<hi.x(); x++ ) {
      
        IntVector c(x,y,z);
      
        c = d_planeAve_1->transformCellIndex(x, y, z);
      
        Qprime[c] = Qlocal[c] - Qbar[z];
        
        //__________________________________
        //  debugging
        if ( c == d_monitorCell && dbg_OTF_MTF.active() ){
          cout << "  calc_Q_prime:  L-"<< L_indx << " " << d_monitorCell <<  setw(10)<< Q->label->getName()
               <<"\t Qprime: "  << Qprime[c] << " Qlocal: " << Qlocal[c] << "\t Q_bar: " << Qbar[z] << endl;
        }
      }
    }
  }
}

//______________________________________________________________________
/*
    iterate over all cells{
      u'u', v'v', w'w'      => CCVariable< Vector > mormalTurbStrss      
      u'v', v'w', w'u'      => CCVariable< Vector > shearTurbStrss

      // scalar
      foreach Q ( T, P, scalar )
        u'Q', v'Q', w'Q'    => CCVariable< Uintah:Vector > Q_turb_flux
      end
    }
*/
//______________________________________________________________________
void meanTurbFluxes::sched_TurbFluxes(SchedulerP   & sched,
                                      const LevelP & level)
{
  Task* t = scinew Task( "meanTurbFluxes::calc_TurbFluxes",
                    this,&meanTurbFluxes::calc_TurbFluxes );

  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::sched_TurbFluxes");
  t->requires( Task::OldDW, m_simulationTimeLabel );
  t->requires( Task::OldDW, d_lb->lastCompTimeLabel );

  Ghost::GhostType gn  = Ghost::None;
  //__________________________________
  //  scalars
  for ( unsigned int i =0 ; i < d_Qvars.size(); i++ ) {
    std::shared_ptr< Qvar > Q = d_Qvars[i];
    t->requires( Task::NewDW, Q->primeLabel, Q->matSubSet, gn, 0 );
    t->computes ( Q->turbFluxLabel );
  }

  //__________________________________
  //  velocity
  t->requires( Task::NewDW, d_velVar->primeLabel, d_velVar->matSubSet, gn, 0 );
  t->computes ( d_velVar->normalTurbStrssLabel );
  t->computes ( d_velVar->shearTurbStrssLabel );

  sched->addTask( t, level->eachPatch() , d_matl_set );
}


//______________________________________________________________________
//
void meanTurbFluxes::calc_TurbFluxes(const ProcessorGroup * ,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * ,
                                     DataWarehouse        * ,
                                     DataWarehouse        * new_dw)
{
  int L_indx = getLevelP( patches )->getIndex();
  
  for( auto p=0;p<patches->size();p++ ){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbg_OTF_MTF, "Doing meanTurbFluxes::calc_TurbFluxes");

    constCCVariable<Vector> velPrime;
    new_dw->get ( velPrime, d_velVar->primeLabel, d_velVar->matl, patch, Ghost::None, 0 );

    //__________________________________
    //  turbulent fluxes Q'u', Q'v', Q'w'
    for ( unsigned int i =0 ; i < d_Qvars.size(); i++ ) {
      std::shared_ptr< Qvar > Q = d_Qvars[i];

      const int matl = Q->matl;

      constCCVariable< double > Qprime;
      new_dw->get ( Qprime, Q->primeLabel, Q->matl, patch, Ghost::None, 0 );

      CCVariable< Vector > QturbFlux;
      new_dw->allocateAndPut( QturbFlux, Q->turbFluxLabel, matl, patch );

      //__________________________________
      //
      for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        QturbFlux[c] = Qprime[c] * velPrime[c];

        //__________________________________
        //  debugging
        if ( c == d_monitorCell && dbg_OTF_MTF.active() ){
          cout << "  calc_TurbFluxes:  L-"<< L_indx << " " << d_monitorCell <<  setw(10)<< Q->label->getName()
               <<"\t QturbFlux: "  << QturbFlux[c] << " Qprime: " << Qprime[c] << "\t velPrime: " << velPrime[c] << endl;
        }
      }
    }    // QVars loop


    //__________________________________
    //   turbulent stresses
    CCVariable< Vector > diag;
    CCVariable< Vector > offdiag;

    new_dw->allocateAndPut( diag,    d_velVar->normalTurbStrssLabel, d_velVar->matl, patch );
    new_dw->allocateAndPut( offdiag, d_velVar->shearTurbStrssLabel,  d_velVar->matl, patch );

    for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      Vector vel = velPrime[c];
      diag[c] = Vector( vel.x() * vel.x(),        // u'u'
                        vel.y() * vel.y(),        // v'v'
                        vel.z() * vel.z() );      // w'w'

      offdiag[c] = Vector( vel.x() * vel.y(),     // u'v'
                           vel.y() * vel.w(),     // v'w'
                           vel.z() * vel.z() );   // w'u'
                           
      //__________________________________
      //  debugging
      if ( c == d_monitorCell && dbg_OTF_MTF.active() ){
        cout << "  calc_TurbFluxes:  L-"<< L_indx << " " << d_monitorCell <<  setw(10)<< d_velVar->label->getName()
             <<"\t diag: "  << diag[c] << " offdiag: " << offdiag[c] << "\t velPrime: " << velPrime[c] << endl;
      }
    }
  }
}
