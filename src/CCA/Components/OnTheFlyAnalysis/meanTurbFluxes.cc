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

#include <CCA/Ports/ApplicationInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Disclosure/TypeUtils.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>
#include <iostream>

using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "meanTurbFluxes:+,planeAverage:+"
Dout dbg_OTF_MTF("meanTurbFluxes", "OnTheFlyAnalysis", "meanTurbFluxes debug stream", false);

//______________________________________________________________________
/*
  ToDo:
    - verification task
______________________________________________________________________*/

meanTurbFluxes::meanTurbFluxes( const ProcessorGroup    * myworld,
                                const MaterialManagerP    materialManager,
                                const ProblemSpecP      & module_spec )
  : AnalysisModule(myworld, materialManager, module_spec)
{
  d_matl_set    = nullptr;
  d_monitorCell = IntVector(0,0,0);

  d_planeAve_1 = scinew planeAverage( myworld, materialManager, module_spec, true,  true, 0);
  d_planeAve_2 = scinew planeAverage( myworld, materialManager, module_spec, false, true,  1);

  d_velVar = make_shared< velocityVar >();
}

//__________________________________
meanTurbFluxes::~meanTurbFluxes()
{
  DOUT(dbg_OTF_MTF, " Doing: destorying meanTurbFluxes" );
  
  if(d_matl_set && d_matl_set->removeReference()) {
    delete d_matl_set;
  }

  delete d_planeAve_1;
  delete d_planeAve_2;
}


//______________________________________________________________________
//  "That C++11 doesn't include make_unique is partly an oversight, and it will
//   almost certainly be added in the future. In the meantime, use the one provided below."
//     - Herb Sutter, chair of the C++ standardization committee
//
//   Once C++14 is adpoted delete this
template<typename T, typename ...Args>
std::unique_ptr<T> meanTurbFluxes::make_unique( Args&& ...args )
{
    return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
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

  // debugging
  m_module_spec->get(    "monitorCell",      d_monitorCell);

  ProblemSpecP vars_ps = m_module_spec->findBlock("Variables");
  if (!vars_ps){
    throw ProblemSetupException("meanTurbFluxes: Couldn't find <Variables> tag", __FILE__, __LINE__);
  }

  //__________________________________
  // Find the material to analyze.  Default is matl 0.
  // The user should specify
  //  <material>   atmosphere </material>
  const Material*  matl;
  if(m_module_spec->findBlock("material") ){
    matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  } else {
    matl = m_materialManager->getMaterial(0);
  }

  int defaultMatl = matl->getDWIndex();

  d_matl_set = d_planeAve_1->d_matl_set;
  
  
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

  const TypeDescription * td_V     = CCVariable<Vector>::getTypeDescription();
  d_velVar->primeLabel            = VarLabel::create( labelName + "_prime",         td_V);  // u', v', w'
  d_velVar->normalTurbStrssLabel  = VarLabel::create( d_velVar->normalTurbStrssName, td_V); // u'u', v'v', w'w' 
  d_velVar->shearTurbStrssLabel   = VarLabel::create( d_velVar->shearTurbStrssName,  td_V); // u'v', v'w', w'u'

  typedef planeAverage PA;
  std::vector< shared_ptr< PA::planarVarBase > >planarVars;

  // create planarAverage variable: normal turbulent stress
  auto pv        = make_shared< PA::planarVar_Vector >();
  pv->label      = d_velVar->normalTurbStrssLabel;
  pv->matl       = defaultMatl;
  pv->level      = ALL_LEVELS;
  pv->baseType   = td_V->getType();
  pv->subType    = TypeDescription::Vector;
  pv->weightType = PA::NCELLS;

  planarVars.push_back( pv );

  // create planarAverage variable: shear turbulent stress
  auto pv2    = make_unique< PA::planarVar_Vector >(*pv);
  pv2->label  = d_velVar->shearTurbStrssLabel;

  planarVars.push_back( move(pv2) );


  //__________________________________
  //  All the scalar variables to be analyzed
  for( ProblemSpecP var_spec = vars_ps->findBlock( "analyze" ); var_spec != nullptr; var_spec = var_spec->findNextBlock( "analyze" ) ) {

    var_spec->getAttributes( attribute );

    //__________________________________
    // label name
    string labelName = attribute["label"];
    VarLabel* label  = VarLabel::find(labelName);
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

    // CC Variables and only doubles
    if(baseType != TypeDescription::CCVariable &&
       subType  != TypeDescription::double_type   ){
      ostringstream warn;
      warn << "ERROR:AnalysisModule:meanTurbFluxes: ("<<label->getName() << " "
           << " only CCVariable<double> variables work" << endl;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    //__________________________________
    // define intermediate quantity label names
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
    auto me           = make_unique< Qvar >(matl);
    me->label         = label;
    me->primeLabel    = primeLabel;
    me->turbFluxLabel = turbFluxLabel;
    d_Qvars.push_back( move(me) );
    
    // planarAve specs
    auto pv        = make_unique< PA::planarVar_Vector >();
    pv->label      = turbFluxLabel;          // u'Q'(y), v'Q'(y), w'Q'(y)
    pv->matl       = matl;
    pv->level      = level;
    pv->baseType   = td_V->getType();
    pv->subType    = TypeDescription::Vector;
    pv->weightType = PA::NCELLS;

    planarVars.push_back( move(pv) );
  }
  
  d_planeAve_2->setAllLevels_planarVars( 0, planarVars );
}

//______________________________________________________________________
//
void meanTurbFluxes::scheduleInitialize(SchedulerP   & sched,
                                        const LevelP & level)
{
  d_planeAve_1->scheduleInitialize( sched, level);
  
  d_planeAve_2->scheduleInitialize( sched, level);
}



//______________________________________________________________________
//
void meanTurbFluxes::scheduleRestartInitialize(SchedulerP   & sched,
                                               const LevelP & level)
{
  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::scheduleRestartInitialize");

  d_planeAve_1->scheduleRestartInitialize( sched, level);
  
  d_planeAve_2->scheduleRestartInitialize( sched, level);
}


//______________________________________________________________________
//
void meanTurbFluxes::scheduleDoAnalysis(SchedulerP   & sched,
                                        const LevelP & level)
{
  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::scheduleDoAnalysis");

  //__________________________________
  // This instantiation of planarAve computes the planar averages of:
  //
  // {u}^bar(y), {v}^bar(y), {w}^bar(y)
  // {Q}^bar(y)   Q = P, T, scalar.....etc  
  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);
  d_planeAve_1->createMPICommunicator( perProcPatches );
  
  d_planeAve_1->sched_computePlanarAve( sched, level );

  d_planeAve_1->sched_writeToFiles(     sched, level, "planeAve" );
  
  d_planeAve_1->sched_resetProgressVar( sched, level );
  
  d_planeAve_1->sched_updateTimeVar(    sched, level );


  //__________________________________
  //  compute u', v', w', Q'
  sched_TurbFluctuations( sched, level );
  
  
  //__________________________________
  //  compute u'u', v'v', w'w'
  //          u'v', v'w', w'u'
  //          u'Q', v'Q', w'Q'
  sched_TurbFluxes(       sched, level );
  
  
  //__________________________________
  // This instantiation of planarAve computes:
  //
  // {u'u'}^bar(y), {v'v'}^bar(y), {w'w'}^bar(y)      => normalTurbStrss
  // {u'v'}^bar(y), {v'w'}^bar(y), {w'u'}^bar(y)      => shearTurbStrss
  // {u'Q'}^bar(y), {v'Q'}^bar(y), {w'Q'}^bar(y)   
  d_planeAve_2->createMPICommunicator( perProcPatches );
  
  d_planeAve_2->sched_computePlanarAve( sched, level );
  
  d_planeAve_2->sched_writeToFiles(     sched, level, "planeAve_TurbFluxes" );
  
  d_planeAve_2->sched_resetProgressVar( sched, level );
  
  d_planeAve_2->sched_updateTimeVar(    sched, level ); 
}

//______________________________________________________________________
/*
    foreach y ( n_planes )
      iterate over all cells in Y plane {
        u' = u - u^bar(y)           Each plane in the grid will have a different *bar value
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
//
void meanTurbFluxes::sched_TurbFluctuations(SchedulerP   & sched,
                                            const LevelP & level)
{
  Task* t = scinew Task( "meanTurbFluxes::calc_TurbFluctuations",
                    this,&meanTurbFluxes::calc_TurbFluctuations );

  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::sched_TurbFluctuations");

  // u,v,w -> u',v',w'
  t->requires( Task::NewDW, d_velVar->label, d_velVar->matSubSet, Ghost::None, 0 );
  t->computes ( d_velVar->primeLabel );
  
  // Q -> Q'
  for ( size_t i =0 ; i < d_Qvars.size(); i++ ) {
    shared_ptr< Qvar > Q = d_Qvars[i];
    t->requires( Task::NewDW, Q->label, Q->matSubSet, Ghost::None, 0 );
    t->computes ( Q->primeLabel );
  }
  sched->addTask( t, level->eachPatch() , d_matl_set, planeAverage::TG_COMPUTE );
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
    for ( size_t i =0 ; i < d_Qvars.size(); i++ ) {
      shared_ptr< Qvar > Q = d_Qvars[i];
      calc_Q_prime< double >( new_dw, patch, Q );
    }

    // u,v,w -> u',v',w'
    calc_Q_prime< Vector >( new_dw, patch, d_velVar);
  }
}
//______________________________________________________________________
//
template <class T>
void meanTurbFluxes::calc_Q_prime( DataWarehouse * new_dw,
                                   const Patch   * patch,
                                   shared_ptr<Qvar> Q)
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
               << setw(10) << "\t Qprime: "  << Qprime[c] 
               << setw(10) << " Qlocal: " << Qlocal[c] << setw(10) << "Q_bar: " << Qbar[z] << endl;
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
//  This is computed every timestep, not necessary
void meanTurbFluxes::sched_TurbFluxes(SchedulerP   & sched,
                                      const LevelP & level)
{
  Task* t = scinew Task( "meanTurbFluxes::calc_TurbFluxes",
                    this,&meanTurbFluxes::calc_TurbFluxes );

  printSchedule(level,dbg_OTF_MTF,"meanTurbFluxes::sched_TurbFluxes");

  Ghost::GhostType gn  = Ghost::None;
  //__________________________________
  //  scalars
  for ( size_t i =0 ; i < d_Qvars.size(); i++ ) {
    shared_ptr< Qvar > Q = d_Qvars[i];
    t->requires( Task::NewDW, Q->primeLabel, Q->matSubSet, gn, 0 );
    t->computes ( Q->turbFluxLabel );
  }

  //__________________________________
  //  velocity
  t->requires( Task::NewDW, d_velVar->primeLabel, d_velVar->matSubSet, gn, 0 );
  t->computes ( d_velVar->normalTurbStrssLabel );
  t->computes ( d_velVar->shearTurbStrssLabel );

  sched->addTask( t, level->eachPatch() , d_matl_set, planeAverage::TG_COMPUTE );
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
    for ( size_t i =0 ; i < d_Qvars.size(); i++ ) {
      shared_ptr< Qvar > Q = d_Qvars[i];

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

//______________________________________________________________________
//
void meanTurbFluxes::sched_computeTaskGraphIndex( SchedulerP& sched,
                                                  const LevelP& level)
{
  d_planeAve_1->sched_computeTaskGraphIndex( sched, level );
}
