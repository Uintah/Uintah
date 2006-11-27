/****************************************
CLASS
    ParticleFieldExtractor

    Visualization control for simulation data that contains
    information on both a regular grid in particle sets.

OVERVIEW TEXT
    This module receives a ParticleGridReader object.  The user
    interface is dynamically created based information provided by the
    ParticleGridReader.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#include "ParticleFieldExtractor.h"

#include <Core/Util/Timer.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticles.h>
#include <Packages/Uintah/Dataflow/Ports/ScalarParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticles.h>
#include <Packages/Uintah/Dataflow/Ports/VectorParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticles.h>
#include <Packages/Uintah/Dataflow/Ports/TensorParticlesPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;
using std::ostringstream;

namespace Uintah {

using namespace SCIRun;

Mutex ParticleFieldExtractor::module_lock("PFEMutex");

DECLARE_MAKER(ParticleFieldExtractor)

//--------------------------------------------------------------- 

ParticleFieldExtractor::ParticleFieldExtractor(GuiContext* ctx) :
  Module("ParticleFieldExtractor", ctx, Filter, "Selectors", "Uintah"),
  tcl_status(get_ctx()->subVar("tcl_status")),
  generation(-1),  timestep(-1), material(-1), levelnum(0),
  level_(get_ctx()->subVar("level")),
  psVar(get_ctx()->subVar("psVar")),
  pvVar(get_ctx()->subVar("pvVar")),
  ptVar(get_ctx()->subVar("ptVar")),
  onMaterials(get_ctx()->subVar("onMaterials")),
  pNMaterials(get_ctx()->subVar("pNMaterials")),
  positionName(""), particleIDs(""), archiveH(0),
  num_materials(0), num_selected_materials(0)
{ 
} 

//------------------------------------------------------------ 
ParticleFieldExtractor::~ParticleFieldExtractor(){} 

//------------------------------------------------------------- 

void
ParticleFieldExtractor::add_type(string &type_list,
                                 const TypeDescription *subtype)
{
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
  case TypeDescription::float_type:
  case TypeDescription::int_type:
    type_list += " scaler";
    break;
  case TypeDescription::Vector:
    type_list += " vector";
    break;
  case TypeDescription::Matrix3:
    type_list += " matrix3";
    break;
  default:
    cerr<<"Error in ParticleFieldExtractor::setVars(): Vartype not implemented.  Aborting process.\n";
    abort();
  }
}  

bool
ParticleFieldExtractor::setVars(DataArchiveHandle& archive, int timestep,
                                bool archive_dirty )
{
  string command;

  names.clear();
  types.clear();
  archive->queryVariables(names, types);

  vector< double > times;
  vector< int > indices;
  archive->queryTimesteps( indices, times );
  GridP grid = archive->queryGrid(times[timestep]);
  int levels = grid->numLevels();
  int guilevel = level_.get();
  LevelP level = grid->getLevel( (guilevel == levels ? levels-1 : guilevel) );
  if( guilevel == levels )  level_.set( levels - 1 );
  Patch* r = *(level->patchesBegin());
  
  // get the number of materials for the particle Variables
  ConsecutiveRangeSet matls;
  int nm = 0;
  for( int i = 0; i < (int)names.size(); i++ ){
    for(int j = 0; j < levels; j++){
      level = grid->getLevel( j );
      for (Level::patchIterator iter = level->patchesBegin();
           (iter != level->patchesEnd()); ++iter){
        matls = archive->queryMaterials(names[i], *iter, times[timestep]);
//         cerr<<"name: "<<names[i]<<", patch: "<<&(*iter)<<", matl_size: "<<matls.size()<<"\n";
        nm = ( (int)matls.size() > nm ? (int)matls.size() : nm );
      }
//       cerr<<"\n";
    }
  }
  ostringstream level_os;

  if( !archive_dirty && nm == num_materials){
    return true;
  } else {
    // make a list of levels with particles on them.  Eventually
    // particles will only exist on one level, but until then
    // Figure out which levels have particles and build the 
    // interface for them. 
    // NOTE: this is not a general solution!
    bool found_particles = false;
    for( int j = 0; j < (int)names.size(); j++ ){
      if( names[j] == "p.x" ){
        for(int i = 0; i < levels; i++){
          level = grid->getLevel( i );
          for (Level::patchIterator iter = level->patchesBegin();
               (iter != level->patchesEnd()); ++iter){
            ParticleVariable<Point> var;
            matls = archive->queryMaterials(names[j], *iter, times[timestep]);
            ConsecutiveRangeSet::iterator it = matls.begin();
            while( it != matls.end() ){
              archive->query( var, names[j], *it, *iter, times[timestep]); 
              if(var.getParticleSet()->numParticles() > 0){
                level_os << i <<" ";
                if( !found_particles ) {
                  found_particles = true; //we have found particles
                }
                break;                  // break out while
              }
              ++it;
            }
            if( found_particles )
              break;  //break out patches loop
          }
          // don't break out of levels loop
        }
        if( found_particles ){
          break;
        }
      }
    }



    //string type_list("");
    //string name_list("");
    scalarVars.clear();
    vectorVars.clear();
    tensorVars.clear();
    pointVars.clear();
    particleIDVar = VarInfo();
  
    //  string ptNames;
  
    //   // reset the vars
    //   psVar.set("");
    //   pvVar.set("");
    //   ptVar.set("");

    string psNames("");
    string pvNames("");
    string ptNames("");
    int psIndex = -1;
    int pvIndex = -1;
    int ptIndex = -1;
    bool psMatches = false;
    bool pvMatches = false;
    bool ptMatches = false;
    // get all of the NC and Particle Variables
    const TypeDescription *td;
    bool found = false;
    for( int i = 0; i < (int)names.size(); i++ ){
      td = types[i];
      if(td->getType() ==  TypeDescription::ParticleVariable){
        const TypeDescription* subtype = td->getSubType();
        matls = archive->queryMaterials(names[i], r, times[timestep]);
        if(matls.size() == 0) continue;
        switch ( subtype->getType() ) {
        case TypeDescription::double_type:
        case TypeDescription::float_type:
        case TypeDescription::int_type:
          scalarVars.push_back(VarInfo(names[i], matls));
          found = true;
          if( psNames.size() != 0 )
            psNames += " ";
          psNames += names[i];
          if(psVar.get() == ""){ psVar.set(names[i].c_str()); }
          if(psVar.get() == names[i].c_str()){
            psMatches = true;
          } else {
            if( psIndex == -1){ psIndex = i; }
          }
          break;
        case  TypeDescription::Vector:
          vectorVars.push_back(VarInfo(names[i], matls));
          found = true;
          if( pvNames.size() != 0 )
            pvNames += " ";
          pvNames += names[i];
          if(pvVar.get() == ""){ pvVar.set(names[i].c_str()); }
          if(pvVar.get() == names[i].c_str()){
            pvMatches = true;
          } else {
            if( pvIndex == -1){ pvIndex = i; }
          }
          break;
        case  TypeDescription::Matrix3:
          tensorVars.push_back(VarInfo(names[i], matls));
          found = true;
          if( ptNames.size() != 0 )
            ptNames += " ";
          ptNames += names[i];
          if(ptVar.get() == ""){ ptVar.set(names[i].c_str()); }
          if(ptVar.get() == names[i].c_str()){
            ptMatches = true;
          } else {
            if( ptIndex == -1){ ptIndex = i; }
          }
          break;
        case  TypeDescription::Point:
          pointVars.push_back(VarInfo(names[i], matls));
          found = true;
          break;
        case TypeDescription::long64_type:
          particleIDVar = VarInfo(names[i], matls);
          found = true;
          break;
        default:
          cerr<<"Unknown particle type\n";
          found = false;
        }// else { Tensor,Other}
      }
    }
  
    if( !psMatches && psIndex != -1 ) {
      psVar.set(names[psIndex].c_str());
    } 
    if( !pvMatches && pvIndex != -1 ) {
      pvVar.set(names[pvIndex].c_str());
    }
    if( !ptMatches && ptIndex != -1 ) {
      ptVar.set(names[ptIndex].c_str());
    }

    // get the number of materials for the NC & particle Variables
    num_materials = nm;
//   cerr << "Number of Materials " << num_materials << endl;

//   cerr<<"selected variables in setVar() are "<<
//     psVar.get()<<" (index "<<psIndex<<"), "<<
//     pvVar.get()<<" (index "<<pvIndex<<"), "<<
//     ptVar.get()<<" (index "<<ptIndex<<")\n";

    string visible;
    get_gui()->eval(get_id() + " isVisible", visible);
    if( visible == "1"){
      get_gui()->execute(get_id() + " destroyFrames");
      get_gui()->execute(get_id() + " build");
      get_gui()->execute(get_id() + " buildLevels "+ level_os.str());
//      get_gui()->execute(get_id() + " buildLevel "+ level_os.str());
      //      get_gui()->execute(get_id() + " setParticleScalars " + psNames.c_str());
      //      get_gui()->execute(get_id() + " setParticleVectors " + pvNames.c_str());
      //      get_gui()->execute(get_id() + " setParticleTensors " + ptNames.c_str());
      get_gui()->execute(get_id() + " buildPMaterials " + to_string(num_materials));
      //      get_gui()->execute(get_id() + " buildVarList");    
    }

    return found;
  }
}


bool
ParticleFieldExtractor::showVarsForMatls()
{
  ConsecutiveRangeSet onMaterials;
  for (int matl = 0; matl < num_materials; matl++) {
     string result;
     get_gui()->eval(get_id() + " isOn p" + to_string(matl), result);
     if ( result == "0")
        continue;
     onMaterials.addInOrder(matl);
  }
  

  bool needToUpdate = false;
  string spNames = getVarsForMaterials(scalarVars, onMaterials, needToUpdate);
  string vpNames = getVarsForMaterials(vectorVars, onMaterials, needToUpdate);
  string tpNames = getVarsForMaterials(tensorVars, onMaterials, needToUpdate);

//   cerr<<"selected variables in showVarsForMatls() are "<<
//     psVar.get()<<" (psVarlist: "<<spNames<<"), "<<
//     pvVar.get()<<" (pvVarlist: "<<vpNames<<"), "<<
//     ptVar.get()<<" (ptVarlist: "<<tpNames<<")\n";
  
  if (needToUpdate) {
    string visible;
    get_gui()->eval(get_id() + " isVisible", visible);
    if( visible == "1"){
      get_gui()->execute(get_id() + " clearVariables");
      get_gui()->execute(get_id() + " setParticleScalars " + spNames.c_str());
      get_gui()->execute(get_id() + " setParticleVectors " + vpNames.c_str());
      get_gui()->execute(get_id() + " setParticleTensors " + tpNames.c_str());
      get_gui()->execute(get_id() + " buildVarList");    
      get_gui()->execute("update idletasks");
      reset_vars(); // ?? what is this for?  // It flushes the cache on varibles - Steve
    }
  }

  if( spNames.size() == 0 && vpNames.size() == 0 && tpNames.size() == 0 )
    return false;

  list<VarInfo>::iterator iter;
  positionName = "";
  for (iter = pointVars.begin(); iter != pointVars.end(); iter++) {
     if (onMaterials.intersected((*iter).matls).size() == onMaterials.size()) {
        positionName = (*iter).name;
        break;
     }
  }
  particleIDs = "";
  if (onMaterials.intersected(particleIDVar.matls).size()
      == onMaterials.size()) {
     particleIDs = particleIDVar.name;
  }
  return true;
}

string
ParticleFieldExtractor::getVarsForMaterials(list<VarInfo>& vars,
                                            const ConsecutiveRangeSet& matls,
                                            bool& needToUpdate)
{
  string names = "";
  list<VarInfo>::iterator iter;
  for (iter = vars.begin(); iter != vars.end(); iter++) {
     if (matls.intersected((*iter).matls).size() == matls.size() &&
         matls.size() != 0 ) {
        names += string(" ") + (*iter).name;
        if (!(*iter).wasShown) {
           needToUpdate = true;
           (*iter).wasShown = true;
        }
     }
     else if ((*iter).wasShown) {
        needToUpdate = true;
        (*iter).wasShown = false;
     }
  }
  
  return names;
}

void
ParticleFieldExtractor::addGraphingVars(long64 particleID,
                                        const list<VarInfo>& vars,
                                        string type)
{
  list<VarInfo>::const_iterator iter;
  int i = 0;
  for (iter = vars.begin(); iter != vars.end(); iter++, i++) {
     ostringstream call;
     call << get_id() << " addGraphingVar " << particleID << " " << (*iter).name <<
        " { " << get_matl_from_particleID(particleID, (*iter).matls) << " } " << type <<
       //       " {" << (*iter).matls.expandedString() << "} " << type <<
        " " << i;
     get_gui()->execute(call.str().c_str());
  }
}

void
ParticleFieldExtractor::callback(long64 particleID)
{
//   cerr<< "ParticleFieldExtractor::callback request data for index "<<
//     particleID << ".\n";

  ostringstream call;
  call << get_id() << " create_part_graph_window " << particleID;
  get_gui()->execute(call.str().c_str());
  addGraphingVars(particleID, scalarVars, "scalar");
  addGraphingVars(particleID, vectorVars, "vector");
  addGraphingVars(particleID, tensorVars, "matrix3");
}               
                
// This may need be made faster in the future.  Right now we are just looping
// over all the particle ID's for each materials searching for a match.
// Since this should only be used in debugging experiments it doesn't need
// to be super speedy, just responsive.
int ParticleFieldExtractor::get_matl_from_particleID(long64 particleID, 
                                                     const ConsecutiveRangeSet& matls) {
  DataArchiveHandle archive = archiveH->getDataArchive();
  GridP grid = archive->queryGrid( time );
  LevelP level = grid->getLevel( 0 );
  // loop over all the materials
  
  for(Level::const_patchIterator patch = level->patchesBegin();
      patch != level->patchesEnd();
      patch++ )
    {
      for(ConsecutiveRangeSet::iterator iter = matls.begin(); 
          iter != matls.end(); ++iter) {
        ParticleVariable< long64 > pvi;
        try {
          archive->query(pvi, particleIDs, *iter, *patch, time);
        } catch(VariableNotFoundInGrid& e) {
          cout << e.message() << "\n";
          continue;
        }
        ParticleSubset* pset = pvi.getParticleSubset();
        // check if we have an particles on this patch
        if(pset->numParticles() > 0){
          // now loop over the ParticleVariables and find it
          for(ParticleSubset::iterator part_iter = pset->begin();
              part_iter != pset->end(); part_iter++) {
            if (pvi[*part_iter] == particleID)
              return *iter;
          }
        }
      }
    }
  // failed to find a matl
  return -1;
}
/*

void
ParticleFieldExtractor::graph(string idx, string var)
{
  int i;
  if( MPParticleGridReader *tpr = dynamic_cast<MPParticleGridReader*> (pgrh.get_rep())){

    Array1<double> values;
    if( tpr->GetNTimesteps() ){
      int varId = tpr->GetParticleSet(pName.get())->find_scalar( var );
      tpr->GetParticleData(atoi(idx()), pName.get(), var,  values);
    
      Array1<double> vs;
      for(i = 0; i < values.size(); i++)
        vs.add( values[i] );
    
      ostringstream ostr;
      ostr << get_id() << " graph " << idx+var<<" "<<var << " ";
      int j = 0;
      for( i = tpr->GetStartTime(); i <= tpr->GetEndTime();
           i += tpr->GetIncrement())
        {
          ostr << i << " " << values[j++] << " ";
        }
      get_gui()->execute( ostr.str().c_str() );
    }
  }
}
*/

//----------------------------------------------------------------

void
ParticleFieldExtractor::execute() 
{ 
  //  const char* old_tag1 = AllocatorSetDefaultTag("ParticleFieldExtractor::execute");
  tcl_status.set("Calling ParticleFieldExtractor!"); 
  //  bool newarchive; RNJ - Not used???
  in = (ArchiveIPort *) get_iport("Data Archive");
  psout = (ScalarParticlesOPort *) get_oport("Scalar Particles");
  pvout = (VectorParticlesOPort *) get_oport("Vector Particles");
  ptout = (TensorParticlesOPort *) get_oport("Tensor Particles");
  ArchiveHandle handle;
   if(!in->get(handle)){
     warning("ParticleFieldExtractor::execute() Didn't get a handle.");
     //     AllocatorSetDefaultTag(old_tag1);
     return;
   }
   
   DataArchiveHandle archive = handle->getDataArchive();

   int new_generation = handle->generation;
   bool archive_dirty = new_generation != generation;
   if( archive_dirty ){
     generation = new_generation;
     // we have a different archive
//      cerr<<"new DataArchive ... \n";
     // empty the cache of stored variables
     material_data_list.clear();
     
     if (archiveH.get_rep()  == 0 ){
       string visible;
       get_gui()->eval(get_id() + " isVisible", visible);
       if( visible == "0" ){
         get_gui()->execute(get_id() + " buildTopLevel");
       }

     }
     
     archiveH = handle;
   }
     
   if( !setVars( archive, archiveH->timestep(), archive_dirty )){
     warning("Cannot read any ParticleVariables, no action.");
     //     AllocatorSetDefaultTag(old_tag1);
     return;
   }

   if( !showVarsForMatls() ) return;
     
   ScalarParticles* sp = 0;
   VectorParticles* vp = 0;
   TensorParticles* tp = 0;

   // what time is it?
   times.clear();
   indices.clear();
   archive->queryTimesteps( indices, times );
   int idx = handle->timestep();
   time = times[idx];

   buildData( archive, time, sp, vp, tp );
   psout->send( sp );
   pvout->send( vp );
   ptout->send( tp );     
   tcl_status.set("Done");
   //   AllocatorSetDefaultTag(old_tag1);
}


void 
ParticleFieldExtractor::buildData(DataArchiveHandle& archive, double time,
                                  ScalarParticles*& sp,
                                  VectorParticles*& vp,
                                  TensorParticles*& tp)
{
  GridP grid = archive->queryGrid( time );
  int levels = grid->numLevels();
  int guilevel = level_.get();
  LevelP level = grid->getLevel( (guilevel == levels ? levels-1 : guilevel) );

 
  PSetHandle pseth = scinew PSet();
  pseth->SetLevel( level );
  pseth->SetCallbackClass( this );
  
  bool have_sp = false;
  bool have_vp = false;
  bool have_tp = false;
  bool have_ids = false;
  int scalar_type = TypeDescription::Unknown;

  for(int i = 0; i < (int)names.size() ; i++) {
    if (names[i] == psVar.get()) {
      scalar_type = types[i]->getSubType()->getType();
    }
  }

  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* sema = scinew Semaphore( "scalar extractor semahpore",
                                      max_workers); 
  Mutex smutex("ScalarParticles Mutex");
  Mutex vmutex("VectorParticles Mutex");
  Mutex tmutex("TensorrParticles Mutex");
  Mutex imutex("ParticleIds Mutex");
//   WallClockTimer my_timer;
//   my_timer.start();
//  double size = level->numPatches();  RNJ - Commented to get rid of unused warning.
//  int count = 0; RNJ - Commented to get rid of unused warning.
  // iterate over patches
  for(Level::const_patchIterator patch = level->patchesBegin();
      patch != level->patchesEnd(); patch++ ){
//     update_progress(count++/size, my_timer);
    sema->down();
    Thread *thrd =
      new Thread( scinew PFEThread( this, archive,
                                    *patch,  sp, vp, tp, pseth,
                                    scalar_type, have_sp, have_vp,
                                    have_tp, have_ids, sema,
                                    &smutex, &vmutex, &tmutex, &imutex, gui_),
                  "Particle Field Extractor Thread");
    thrd->detach();
//     PFEThread *thrd = scinew PFEThread( this, archive, *patch,
//                           sp, vp, tp, pseth,
//                           scalar_type, have_sp, have_vp,
//                           have_tp, have_ids, sema,
//                           &smutex, &vmutex, &tmutex, &imutex, gui);

//     thrd->run(); 
  }
  sema->down( max_workers );
  if( sema )  delete sema;
//   timer.add( my_timer.time() );
//   my_timer.stop();
} 
void
PFEThread::run()
{
  ParticleSubset* dest_subset = scinew ParticleSubset();
  ParticleVariable< long64 > ids( dest_subset );
  ParticleVariable< Vector > vectors(dest_subset);
  ParticleVariable< Point > positions(dest_subset);
  ParticleVariable< double > scalars(dest_subset);
  ParticleVariable< Matrix3 > tensors( dest_subset );

  ParticleVariable< Vector > pvv;
  ParticleVariable< Matrix3 > pvt;
  ParticleVariable< double > pvs;
  ParticleVariable< float > pvfloat;
  ParticleVariable< Point  > pvp;
  ParticleVariable< int > pvint;
  ParticleVariable< long64 > pvi;

  //int numMatls = 29;

  for(int matl = 0; matl < pfe->num_materials; matl++) {
    string result;
    ParticleSubset* source_subset = 0;
    bool have_subset = false;

    gui->eval(pfe->get_id() + " isOn p" + to_string(matl), result);
    if ( result == "0")
      continue;
    if (pfe->pvVar.get() != ""){
      have_vp = true;
      archive->query(pvv, pfe->pvVar.get(), matl, patch, pfe->time);    
      if( !have_subset){
        source_subset = pvv.getParticleSubset();
        have_subset = true;
      }
    }
    if( pfe->psVar.get() != ""){
      have_sp = true;
      switch (scalar_type) {
      case TypeDescription::double_type:
        archive->query(pvs, pfe->psVar.get(), matl, patch, pfe->time);
        if( !have_subset){
          source_subset = pvs.getParticleSubset();
          have_subset = true;
        }
        break;
      case TypeDescription::float_type:
        //cerr << "Getting data for ParticleVariable<float>\n";
        archive->query(pvfloat, pfe->psVar.get(), matl, patch, pfe->time);
        if( !have_subset){
          source_subset = pvfloat.getParticleSubset();
          have_subset = true;
        }
        //cerr << "Got data\n";
        break;
      case TypeDescription::int_type:
        //cerr << "Getting data for ParticleVariable<int>\n";
        archive->query(pvint, pfe->psVar.get(), matl, patch, pfe->time);
        if( !have_subset){
          source_subset = pvint.getParticleSubset();
          have_subset = true;
        }
        //cerr << "Got data\n";
        break;
      }
    }
    if (pfe->ptVar.get() != ""){
      have_tp = true;
      archive->query(pvt, pfe->ptVar.get(), matl, patch, pfe->time);
      if( !have_subset){
        source_subset = pvt.getParticleSubset();
        have_subset = true;
      }
    }
    if(pfe->positionName != "")
      archive->query(pvp, pfe->positionName, matl, patch, pfe->time);

    if(pfe->particleIDs != ""){
      //cerr<<"paricleIDs = "<<pfe->particleIDs<<endl;
      have_ids = true;
      archive->query(pvi, pfe->particleIDs, matl, patch, pfe->time);
    }

    if( !have_subset ){
      sema->up();
      return;
    }

    string elems;
    //unsigned long totsize;
    //void* mem_start;
    //        cerr<<"material "<< matl <<".\n";

//     cerr<<"source_subset has "<<source_subset->numParticles() <<" particles\n";
    particleIndex dest = dest_subset->addParticles(source_subset->numParticles());

    //      cerr<<"dest_subset has "<<dest_subset->numParticles() <<" particles\n";
    //     pvs.getSizeInfo(elems, totsize, mem_start );
    //     cerr<<"there are "<<elems<<" scalar elements for patch "<<patchn
    //  <<"  mat "<<matl<<endl;
    //     pvv.getSizeInfo(elems, totsize, mem_start );
    //     cerr<<"there are "<<elems<<" vector elements for patch "<<patchn
    //  <<"  mat "<<matl<<endl;
    //     pvt.getSizeInfo(elems, totsize, mem_start );
    //     cerr<<"there are "<<elems<<" tensor elements for patch "<<patchn
    //  <<"  mat "<<matl<<endl;
      
    vectors.resync();
    positions.resync();
    ids.resync();
    scalars.resync();
    tensors.resync();

    for(ParticleSubset::iterator iter = source_subset->begin();
        iter != source_subset->end(); iter++, dest++){
      if(have_vp)
        vectors[dest]=pvv[*iter];
      else
        vectors[dest]=Vector(0,0,0);
      if(have_sp)
        switch (scalar_type) {
        case TypeDescription::double_type:
          scalars[dest]=pvs[*iter];
          break;
        case TypeDescription::float_type:
          scalars[dest]=pvfloat[*iter];
          break;
        case TypeDescription::int_type:
          scalars[dest]=pvint[*iter];
          break;
        }
      else
        scalars[dest]=0;
      if(have_tp){
        tensors[dest]=pvt[*iter];
      }
      else
        tensors[dest]=Matrix3(0.0);
      if(have_ids)
        ids[dest] = pvi[*iter];
      else
        ids[dest] = PARTICLE_FIELD_EXTRACTOR_BOGUS_PART_ID;
      
      positions[dest]=pvp[*iter];
    }
  }
  imutex->lock();
  pseth->AddParticles( positions, ids, patch);
  imutex->unlock();
  if(have_sp) {
    smutex->lock();
    if( sp == 0 ){
      sp = scinew ScalarParticles();
      sp->Set( pseth );
    }
    sp->AddVar( scalars );
    smutex->unlock();
  } else 
    sp = 0;
  if(have_vp) {
    vmutex->lock();
    if( vp == 0 ){
      vp = scinew VectorParticles();
      vp->Set( pseth );
    }
    vp->AddVar( vectors );
    vmutex->unlock();
  } else 
    vp = 0;

  if(have_tp){
    tmutex->lock();
    if( tp == 0 ){
      tp = scinew TensorParticles();
      tp->Set( pseth );
    }
    tp->AddVar( tensors);
    tmutex->unlock();
  } else
    tp = 0;

  sema->up();
}


void
ParticleFieldExtractor::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  if(args[1] == "extract_data") {
    int i = 2;
    string displaymode(args[i++]);
    string varname(args[i++]);
    string particleID(args[i++]);
    int num_mat;
    string_to_int(args[i++], num_mat);
//     cerr << "Extracting " << num_mat << " materals:";
    vector< string > mat_list;
    vector< string > type_list;
    for (int j = i; j < i+(num_mat*2); j++) {
      string mat(args[j]);
      mat_list.push_back(mat);
      j++;
      string type(args[j]);
      type_list.push_back(type);
    }
//     cerr << endl;
//     cerr << "Graphing " << varname << " with materials: " << vector_to_string(mat_list) << endl;
    extract_data(displaymode,varname,mat_list,type_list,particleID);
  } 
  else {
    Module::tcl_command(args, userdata);
  }

}

void
ParticleFieldExtractor::extract_data(string display_mode,
                                     string varname, 
                                     vector<string> mat_list,
                                     vector<string> type_list, 
                                     string particleID)
{

  /* void DataArchive::query(std::vector<T>& values, const std::string& name,
                        int matlIndex, long particleID,
                        double startTime, double endTime);
  */
  // clear the current contents of the ticles's material data list
  get_gui()->execute(get_id() + " reset_var_val");

  // determine type
  const TypeDescription *td = 0;
  for(int i = 0; i < (int)names.size() ; i++)
    if (names[i] == varname)
      td = types[i];

  if (td == 0) {
    // You are in some serious trouble
    error("ParticleFieldExtractor::graph::Type for specified variable is not found");
    return;
  }
  DataArchiveHandle archive = this->archiveH->getDataArchive();
  vector< int > indices;
  times.clear();
  archive->queryTimesteps( indices, times );
  get_gui()->execute(get_id() + " setTime_list " + vector_to_string(indices).c_str());

  string name_list("");
#ifdef _WIN32
  // try it this way, we don't have atoll
  istringstream is(particleID);
  long64 partID;
  is >> partID;
#else
  long64 partID = atoll(particleID.c_str());
#endif
//   cout << "partID = "<<partID<<endl;
//   cerr << "mat_list.size() = "<<mat_list.size()<<endl;
  for(int m = 0; m < (int)mat_list.size(); m++) {
//     cerr << "mat_list["<<m<<"] = "<<mat_list[m]<<endl;
  }
  const TypeDescription* subtype = td->getSubType();
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
    cerr << "Graphing a variable of type double\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      // query the value
      if (!is_cached(particleID+" "+varname+" "+mat_list[i],data)) {
        // query the value and then cache it
        vector< double > values;
        int matl = atoi(mat_list[i].c_str());
//         cerr << "querying data archive for "<<varname<<" with matl="<<matl<<", particleID="<<partID<<", from time "<<times[0]<<" to time "<<times[times.size()-1]<<endl;
        try {
          archive->query(values, varname, matl, partID, times[0], times[times.size()-1]);
        } catch (const VariableNotFoundInGrid& exception) {
          error("Particle Variable "+particleID+" not found.\n");
          return;
        }
//      cerr << "Received data.  Size of data = " << values.size() << endl;
        cache_value(particleID+" "+varname+" "+mat_list[i],values,data);
      } else {
//      cerr << "Cache hit\n";
      }
      get_gui()->execute(get_id()+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::float_type:
//     cerr << "Graphing a variable of type float\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      // query the value
      if (!is_cached(particleID+" "+varname+" "+mat_list[i],data)) {
        // query the value and then cache it
        vector< float > values;
        int matl = atoi(mat_list[i].c_str());
//      cerr << "querying data archive for "<<varname<<" with matl="<<matl<<", particleID="<<partID<<", from time "<<times[0]<<" to time "<<times[times.size()-1]<<endl;
        try {
          archive->query(values, varname, matl, partID, times[0], times[times.size()-1]);
        } catch (const VariableNotFoundInGrid& exception) {
          error("Particle Variable "+particleID+" not found.\n");
          return;
        }
//      cerr << "Received data.  Size of data = " << values.size() << endl;
        cache_value(particleID+" "+varname+" "+mat_list[i],values,data);
      } else {
//      cerr << "Cache hit\n";
      }
      get_gui()->execute(get_id()+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::int_type:
//     cerr << "Graphing a variable of type int\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      // query the value
      if (!is_cached(particleID+" "+varname+" "+mat_list[i],data)) {
        // query the value and then cache it
        vector< int > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive->query(values, varname, matl, partID, times[0], times[times.size()-1]);
        } catch (const VariableNotFoundInGrid& exception) {
          error("Particle Variable "+particleID+" not found.\n");
          return;
        }
//      cerr << "Received data.  Size of data = " << values.size() << endl;
        cache_value(particleID+" "+varname+" "+mat_list[i],values,data);
      } else {
//      cerr << "Cache hit\n";
      }
      get_gui()->execute(get_id()+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Vector:
//     cerr << "Graphing a variable of type Vector\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(particleID+" "+varname+" "+mat_list[i]+" "+type_list[i],
                     data)) {
        // query the value
        vector< Vector > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive->query(values, varname, matl, partID, times[0], times[times.size()-1]);
        } catch (const VariableNotFoundInGrid& exception) {
          error("Particle Variable "+particleID+" not found.\n");
          return;
        }
//      cerr << "Received data.  Size of data = " << values.size() << endl;
        data = vector_to_string(values,type_list[i]);
        cache_value(particleID+" "+varname+" "+mat_list[i],values);
      } else {
//      cerr << "Cache hit\n";
      }
      get_gui()->execute(get_id()+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";      
    }
    break;
  case TypeDescription::Matrix3:
//     cerr << "Graphing a variable of type Matrix3\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(particleID+" "+varname+" "+mat_list[i]+" "+type_list[i],
                     data)) {
        // query the value
        vector< Matrix3 > values;
        int matl = atoi(mat_list[i].c_str());
        try {
          archive->query(values, varname, matl, partID, times[0], times[times.size()-1]);
        } catch (const VariableNotFoundInGrid& exception) {
          error("Particle Variable "+particleID+" not found.\n");
          return;
        }
//      cerr << "Received data.  Size of data = " << values.size() << endl;
        data = vector_to_string(values,type_list[i]);
        cache_value(particleID+" "+varname+" "+mat_list[i],values);
      } else {
//      cerr << "Cache hit\n";
      }
      get_gui()->execute(get_id()+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";      
    }
    break;
  default:
    error("Unknown var type");
    return;
  }// else { Tensor,Other}
  //cerr << "callig graph_data with \"particleID="<<particleID<<" varname="<<varname<<" name_list="<<name_list<<endl;
  get_gui()->execute(get_id()+" "+display_mode.c_str()+"_data "+particleID.c_str()+" "
               +varname.c_str()+" "
               +name_list.c_str());

}

string
ParticleFieldExtractor::vector_to_string(vector< int > data)
{
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string
ParticleFieldExtractor::vector_to_string(vector< string > data)
{
  string result;
  for(int i = 0; i < (int)data.size(); i++) {
      result+= (data[i] + " ");
    }
  return result;
}

string
ParticleFieldExtractor::vector_to_string(vector< double > data)
{
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string
ParticleFieldExtractor::vector_to_string(vector< float > data)
{
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string
ParticleFieldExtractor::vector_to_string(vector< Vector > data, string type)
{
  ostringstream ostr;
  if (type == "length") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length() << " ";
    }
  } else if (type == "length2") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length2() << " ";
    }
  } else if (type == "x") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].x() << " ";
    }
  } else if (type == "y") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].y() << " ";
    }
  } else if (type == "z") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].z() << " ";
    }
  }

  return ostr.str();
}

string
ParticleFieldExtractor::vector_to_string(vector< Matrix3 > data, string type)
{
  ostringstream ostr;
  if (type == "Determinant") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Determinant() << " ";
    }
  } else if (type == "Trace") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Trace() << " ";
    }
  } else if (type == "Norm") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Norm() << " ";
    } 
 }

  return ostr.str();
}

bool
ParticleFieldExtractor::is_cached(string name, string& data)
{
  map< string, string >::iterator iter;
  iter = material_data_list.find(name);
  if (iter == material_data_list.end()) {
    return false;
  }
  else {
    data = iter->second;
    return true;
  }
}

void
ParticleFieldExtractor::cache_value(string where, vector<double>& values,
                                    string &data)
{
  data = vector_to_string(values);
  material_data_list[where] = data;
}

void ParticleFieldExtractor::cache_value(string where, vector<float>& values,
                                         string &data)
{
  data = vector_to_string(values);
  material_data_list[where] = data;
}

void
ParticleFieldExtractor::cache_value(string where, vector<int>& values,
                                    string &data)
{
  data = vector_to_string(values);
  material_data_list[where] = data;
}

void
ParticleFieldExtractor::cache_value(string where, vector<Vector>& values)
{
  string data = vector_to_string(values,"length");
  material_data_list[where+" length"] = data;
  data = vector_to_string(values,"length2");
  material_data_list[where+" length2"] = data;
  data = vector_to_string(values,"x");
  material_data_list[where+" x"] = data;
  data = vector_to_string(values,"y");
  material_data_list[where+" y"] = data;
  data = vector_to_string(values,"z");
  material_data_list[where+" z"] = data;
}

void
ParticleFieldExtractor::cache_value(string where, vector<Matrix3>& values)
{
  string data = vector_to_string(values,"Determinant");
  material_data_list[where+" Determinant"] = data;
  data = vector_to_string(values,"Trace");
  material_data_list[where+" Trace"] = data;
  data = vector_to_string(values,"Norm");
  material_data_list[where+" Norm"] = data;
}

} // End namespace Uintah
 
