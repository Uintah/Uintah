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
#include <fstream>

#define ALL_LEVELS 99

using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output
//  setenv SCI_DEBUG "meanTurbFluxes:+,planeAverage:+"
Dout dbg_OTF_MTF("meanTurbFluxes", "OnTheFlyAnalysis", "meanTurbFluxes debug stream", false);

//______________________________________________________________________
/*
Verification steps:
    1)  Edit: 
        src/CCA/Components/OnTheFlyAnalysis/meanTurbFluxesVerify.py
        nPlaneCells   = np.array( [30, 30, 1]) 
        change nPlaneCells to match your problem.
        
    2)  Execute: meanTurbFluxesVerify.py
        To generate the files "testDistribution.txt" and "covariance.txt"
        
    3) Modify the labels used in meanTurbFluxes section
            <velocity label="verifyVelocity" />
            <analyze label="verifyScalar"     weighting="nCells" /> 
            <analyze label="verifyVelocity"   weighting="nCells" />
       
       Also turn on verification task:
        <enableVerificationTask/>
     
     4) Run the simulation and compare:
         <uda>/TurbFluxes/<timestep>/L-0/verifyScalar_turbFlux_0, normalTurbStrss_0,  shearTurbStrss_0
        
        to
        "covariance.txt"
          
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

  d_lastCompTimeLabel = d_planeAve_1->d_lb->lastCompTimeLabel;
  d_velocityVar = make_shared< velocityVar >();

  d_verifyScalarLabel=VarLabel::create( "verifyScalar",   CCVariable<double>::getTypeDescription() );
  d_verifyVectorLabel=VarLabel::create( "verifyVelocity", CCVariable<Vector>::getTypeDescription() );
}

//__________________________________
meanTurbFluxes::~meanTurbFluxes()
{
  DOUT(dbg_OTF_MTF, " Doing: destorying meanTurbFluxes" );

  delete d_planeAve_1;
  delete d_planeAve_2;

  VarLabel::destroy( d_verifyVectorLabel );
  VarLabel::destroy( d_verifyScalarLabel );
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
  //  enable Verification
  if( m_module_spec->findBlock( "enableVerification")  !=  nullptr ) {
    d_doVerification = true;
  }

  //__________________________________
  //  velocity label
  map<string,string> attribute;
  ProblemSpecP vel_ps = m_module_spec->findBlock( "velocity" );

  if( vel_ps == nullptr ) {
    throw ProblemSetupException("meanTurbFluxes: velocity xml tag not found: ", __FILE__, __LINE__);
  }

  vel_ps->getAttributes( attribute );
  string labelName = attribute["label"];
  d_velocityVar->label   = VarLabel::find( labelName );

  if( d_velocityVar->label == nullptr ){
    throw ProblemSetupException("meanTurbFluxes: velocity label not found: " + labelName , __FILE__, __LINE__);
  }

  d_velocityVar->matl  = defaultMatl;

  const TypeDescription * td_V     = CCVariable<Vector>::getTypeDescription();
  d_velocityVar->primeLabel            = VarLabel::create( labelName + "_prime",         td_V);  // u', v', w'
  d_velocityVar->normalTurbStrssLabel  = VarLabel::create( d_velocityVar->normalTurbStrssName, td_V); // u'u', v'v', w'w'
  d_velocityVar->shearTurbStrssLabel   = VarLabel::create( d_velocityVar->shearTurbStrssName,  td_V); // u'v', v'w', w'u'

  typedef planeAverage PA;
  std::vector< shared_ptr< PA::planarVarBase > >planarVars;

  // create planarAverage variable: normal turbulent stress
  auto pv        = make_shared< PA::planarVar_Vector >();
  pv->label      = d_velocityVar->normalTurbStrssLabel;
  pv->matl       = defaultMatl;
  pv->level      = ALL_LEVELS;
  pv->baseType   = td_V->getType();
  pv->subType    = TypeDescription::Vector;
  pv->weightType = PA::NCELLS;
  pv->fileDesc   = "u'u'__________________v'v'______________w'w'";

  planarVars.push_back( pv );

  // create planarAverage variable: shear turbulent stress
  auto pv2    = make_unique< PA::planarVar_Vector >(*pv);  // this is a deep copy
  pv2->label  = d_velocityVar->shearTurbStrssLabel;
  pv2->fileDesc   = "u'v'__________________v'w'______________w'u'";

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

    if (label == d_velocityVar->label ){  // velocity label has already been processed
      continue;
    }

    //__________________________________
    //  bulletproofing
    const TypeDescription* td      = label->typeDescription();
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

    // Bulletproofing 
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
    pv->fileDesc   = "u'Q'_________________v'Q'__________________w'Q'";

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


  sched_populateVerifyLabels( sched, level );

  //__________________________________
  // This instantiation of planarAve computes the planar averages of:
  //
  // {u}^bar(y), {v}^bar(y), {w}^bar(y)
  // {Q}^bar(y)   Q = P, T, scalar.....etc
  const PatchSet* perProcPatches = m_scheduler->getLoadBalancer()->getPerProcessorPatchSet(level);
  d_planeAve_1->createMPICommunicator( perProcPatches );

  d_planeAve_1->sched_computePlanarAve( sched, level );

  d_planeAve_1->sched_writeToFiles(     sched, level, "planarAve" );

  d_planeAve_1->sched_resetProgressVar( sched, level );

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

  d_planeAve_2->sched_writeToFiles(     sched, level, "TurbFluxes" );

  d_planeAve_2->sched_resetProgressVar( sched, level );
}

//______________________________________________________________________
//  This task reads a file containing multivariant normal distribution
//  and fills each plane with these values.  The values are duplicated
//  on all other planes on all patches
void meanTurbFluxes::sched_populateVerifyLabels( SchedulerP   & sched,
                                                 const LevelP & level )
{
  if( ! d_doVerification ){
    return;
  }
  
  Task* t = scinew Task( "meanTurbFluxes::populateVerifyLabels",
                    this,&meanTurbFluxes::populateVerifyLabels );

  t->computes ( d_verifyVectorLabel );
  t->computes ( d_verifyScalarLabel );
  sched->addTask( t, level->eachPatch() , d_matl_set );
}
//______________________________________________________________________
//
void meanTurbFluxes::populateVerifyLabels(const ProcessorGroup * pg,
                                          const PatchSubset    * patches,
                                          const MaterialSubset * ,
                                          DataWarehouse        * ,
                                          DataWarehouse        * new_dw)
{



  for( auto p=0;p<patches->size();p++ ){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch, dbg_OTF_MTF, "Doing meanTurbFluxes::verification");

    int matl = d_velocityVar->matl;
    CCVariable< Vector > velocity;
    new_dw->allocateAndPut( velocity, d_verifyVectorLabel, matl, patch );

    CCVariable< double > scalar;
    new_dw->allocateAndPut( scalar,  d_verifyScalarLabel, matl, patch );

    //__________________________________
    // Open the file
    string filename = "testDistribution.txt";
    ifstream ifs;

    ifs.open ( filename, ifstream::in );
    if ( !ifs.good() ){
      string warn = "ERROR opening verification file " + filename;
      throw InternalError( warn, __FILE__, __LINE__ );
    }

    string line;            // row from the file
    int fpos = ifs.tellg(); // file fposition

    //__________________________________
    //  ignore header lines (#)
    while ( getline( ifs, line ) ){
      if ( line[0] != '#'){
        break;
      }
      fpos = ifs.tellg();
    }

    // rewind one line
    ifs.seekg( fpos );

    //__________________________________
    //  find number of cells in the plane of interest on this patch
    IntVector pLo;         // plane lo and hi
    IntVector pHi;
    GridIterator iter = patch->getCellIterator();
    d_planeAve_1->planeIterator( iter, pLo, pHi );

    int nPlaneCellsPerPatch = ( pHi.x() - pLo.x() ) * ( pHi.y() - pLo.y() );

    int lineNum = findFilePositionOffset( patches, nPlaneCellsPerPatch, pLo, pHi);

    //__________________________________
    //  bulletproofing
    int nFileLines = 0;

    while ( getline(ifs, line, '\n') ){
      ++nFileLines;
    }

    if( lineNum > nFileLines ){
      ostringstream warn;
      warn << "\n\nERROR:  The filePosition ("<< lineNum << ") exceeds the length of the "
           << " verification file ("<< filename << ":"<< nFileLines << ").\n"
           << " Verify that the meanTurbFluxesVerify.py:nPlaneCells variable"
           << " matches the ups resolution.\n";
      throw InternalError( warn.str(), __FILE__, __LINE__ );
    }

    //__________________________________
    //  move file fpositon forward (lineNum)
    ifs.clear();        // rewind file fposition
    ifs.seekg( fpos );

    int l = 0;
    
    while ( l != lineNum){
      getline(ifs, line, '\n');
      l++;
    }
    fpos= ifs.tellg();
//    DOUT( true, pg->myRank() << " patch: " << patch->getID() << " lineNum: " << lineNum << " line:"  << line);


    //__________________________________
    //  Loop over the cells in first plane on this patch and read in the
    //  entries from the file
    //    #   u,    v,    w,   scalar
    int plane0_idx = pLo.z();                                  // loop over all cells in plane0
    for ( auto y = pLo.y(); y<pHi.y(); y++ ) {
      for ( auto x = pLo.x(); x<pHi.x(); x++ ) {

        IntVector c;
        c  = d_planeAve_1->transformCellIndex(x, y, plane0_idx);

        // load row into a stringstream
        getline( ifs, line );
        stringstream ss;
        ss << line;

        // Load each value of the row into num_str
        std::vector<std::string> num_str;

        for (int col = 0; col < 5; col++){
          std::string str;
          getline(ss, str, ',');
          num_str.push_back( str );
        }

        // convert from str -> int/double
        //int l    = stoi( num_str[0] );
        double u = stod( num_str[1] );
        double v = stod( num_str[2] );
        double w = stod( num_str[3] );
        double s = stod( num_str[4] );

        velocity[c] = Vector( u, v, w );
        scalar[c]   = s;

       // printf( "%i %i, %15.16e, %15.16e, %15.16e, %15.16e\n",pg->myRank(), l,u,v,w,s );
      }
    }

    //__________________________________
    //  copy the values from first plane on this patch to the
    //  remaining planes.
    for ( auto planeN_idx = pLo.z()+1; planeN_idx<pHi.z(); planeN_idx++ ) {         // loop over the planes in this patch
     // int l = lineNum;

      for ( auto y = pLo.y(); y<pHi.y(); y++ ) {
        for ( auto x = pLo.x(); x<pHi.x(); x++ ) {
          IntVector p0;
          IntVector cur;
          
          p0  = d_planeAve_1->transformCellIndex(x, y, plane0_idx);
          cur = d_planeAve_1->transformCellIndex(x, y, planeN_idx);

          velocity[cur] =  velocity[p0];
          scalar[cur]   =  scalar[p0];
        #if 0
          printf( "%i, %i %15.16e, %15.16e, %15.16e, %15.16e\n",pg->myRank(),
               l,
               velocity[cur].x(),
               velocity[cur].y(),
               velocity[cur].z(),
               scalar[cur] );
          l++;
        #endif
        }
      }
    }
  }
}


//______________________________________________________________________
//  Return the number of lines after the header
//
//  This is tricky!
//  
//  Algorithm;
//    1) Create a map that contains the patchID and file offset for
//       all patches containing the 0th plane.  
//
//    2) For find the equivalent patch that contains plane 0, based on the transformed
//       patch low index.
//    
//    3) Look in the map for the offset of the equivalent patch
//  WARNING:  This could be slow on large core count simulations
//______________________________________________________________________
int
meanTurbFluxes::findFilePositionOffset( const PatchSubset  * patches,
                                        const int nPlaneCellPerPatch,
                                        const IntVector      pLo,
                                        const IntVector      pHi)
{
  map<int, int> fileOffsetMap;

  const LevelP level = getLevelP( patches );

  //__________________________________
  // Find patches that contain the 0th plane
  // and store the patch ID in a map
  int  nPlanePatch = 0;

  for(Level::const_patch_iterator iter=level->patchesBegin(); iter < level->patchesEnd(); iter++) {

    const Patch* patch = *iter;
    IntVector lo = patch->getCellLowIndex();

    bool is0th_planePatch = false;

    // Is the lo index for this plane == 0?
    switch( d_planeAve_1->d_planeOrientation ){

      case planeAverage::XY:{                 // z is constant
        is0th_planePatch = ( lo.z() == 0 );
        break;
      }
      case planeAverage::XZ:{                 // y is constant
        is0th_planePatch = ( lo.y() == 0 );
        break;
      }
      case planeAverage::YZ:{                 // x is constant
        is0th_planePatch = ( lo.x() == 0 );
        break;
      }
      default:
        break;
    }

    //__________________________________
    //  compute the file offset
    if( is0th_planePatch ){
      int offset = nPlanePatch * nPlaneCellPerPatch;
      int id     = patch->getID();
      fileOffsetMap[id] = offset;
      nPlanePatch += 1;
    }
  }

  //__________________________________
  //  Find the equivalent patch containing plane0
  //  pLo.z() = 0 is on plane0
  IntVector plane0_cell     = d_planeAve_1->transformCellIndex( pLo.x(), pLo.y(), 0 );
  
  const Patch* plane0_patch = level->getPatchFromIndex( plane0_cell, false );
  int id_p0 = plane0_patch->getID();

  return fileOffsetMap[id_p0];
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

  sched_TimeVars( t, level, d_lastCompTimeLabel, false );

  // u,v,w -> u',v',w'
  t->requires( Task::NewDW, d_velocityVar->label, d_velocityVar->matSubSet, Ghost::None, 0 );
  t->computes ( d_velocityVar->primeLabel );

  // Q -> Q'
  for ( size_t i =0 ; i < d_Qvars.size(); i++ ) {
    shared_ptr< Qvar > Q = d_Qvars[i];
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
                                           DataWarehouse        * old_dw,
                                           DataWarehouse        * new_dw)
{
  const Level* level = getLevel(patches);
  if( d_planeAve_1->isItTime( old_dw, level, d_lastCompTimeLabel) == false ){
    return;
  }

 //__________________________________
  for( auto p=0;p<patches->size();p++ ){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbg_OTF_MTF, "Doing meanTurbFluxes::calc_TurbFluctuations");

    // Q -> Q'
    for ( size_t i =0 ; i < d_Qvars.size(); i++ ) {
      shared_ptr< Qvar > Q = d_Qvars[i];
      calc_Q_prime< double >( new_dw, patch, Q );
    }

    // u,v,w -> u',v',w'
    calc_Q_prime< Vector >( new_dw, patch, d_velocityVar);
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

  // bulletproofing
  if( Qbar.size() == 0 ){
    string name = Q->label->getName();
    ostringstream err;
    err << "\n\tERROR meanTurbFluxes::calc_Q_prime.  Could not find the planarAverage"
        << " for the variable (" << name << ")."  
        << " \n\t" << name << " must be one of the variables"
        << " listed in the ups file: <DataAnalysis>-><Module name=\"meanTurbFluxes\">-><Variables>-><analyze>\n";
    throw InternalError( err.str(), __FILE__, __LINE__ );
  }

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


  sched_TimeVars( t, level, d_lastCompTimeLabel, false );

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
  t->requires( Task::NewDW, d_velocityVar->primeLabel, d_velocityVar->matSubSet, gn, 0 );
  t->computes ( d_velocityVar->normalTurbStrssLabel );
  t->computes ( d_velocityVar->shearTurbStrssLabel );

  sched->addTask( t, level->eachPatch() , d_matl_set );
}


//______________________________________________________________________
//
void meanTurbFluxes::calc_TurbFluxes(const ProcessorGroup * ,
                                     const PatchSubset    * patches,
                                     const MaterialSubset * ,
                                     DataWarehouse        * old_dw,
                                     DataWarehouse        * new_dw)
{
  const Level* level = getLevel(patches);
  int L_indx = level->getIndex();

  if( d_planeAve_1->isItTime( old_dw, level, d_lastCompTimeLabel) == false ){
    return;
  }

  for( auto p=0;p<patches->size();p++ ){
    const Patch* patch = patches->get(p);
    
    printTask(patches, patch, dbg_OTF_MTF, "Doing meanTurbFluxes::calc_TurbFluxes");

    constCCVariable<Vector> velPrime;
    new_dw->get ( velPrime, d_velocityVar->primeLabel, d_velocityVar->matl, patch, Ghost::None, 0 );

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

    new_dw->allocateAndPut( diag,    d_velocityVar->normalTurbStrssLabel, d_velocityVar->matl, patch );
    new_dw->allocateAndPut( offdiag, d_velocityVar->shearTurbStrssLabel,  d_velocityVar->matl, patch );

    for (CellIterator iter=patch->getCellIterator();!iter.done();iter++){
      IntVector c = *iter;
      
      Vector vel = velPrime[c];
      diag[c] = Vector( vel.x() * vel.x(),        // u'u'
                        vel.y() * vel.y(),        // v'v'
                        vel.z() * vel.z() );      // w'w'

      offdiag[c] = Vector( vel.x() * vel.y(),     // u'v'
                           vel.y() * vel.w(),     // v'w'
                           vel.z() * vel.u() );   // w'u'

      //__________________________________
      //  debugging
      if ( c == d_monitorCell && dbg_OTF_MTF.active() ){
        cout << "  calc_TurbFluxes:  L-"<< L_indx << " " << d_monitorCell <<  setw(10)<< d_velocityVar->label->getName()
             <<"\t diag: "  << diag[c] << " offdiag: " << offdiag[c] << "\t velPrime: " << velPrime[c] << endl;
      }
    }
  }
}
