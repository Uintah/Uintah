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

#include <Core/Util/NotFinished.h>
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
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace Uintah {


using namespace SCIRun;

Mutex ParticleFieldExtractor::module_lock("PFEMutex");

  DECLARE_MAKER(ParticleFieldExtractor)

//--------------------------------------------------------------- 
  ParticleFieldExtractor::ParticleFieldExtractor(GuiContext* ctx)
  : Module("ParticleFieldExtractor", ctx, Filter, "Selectors", "Uintah"),
    tcl_status(ctx->subVar("tcl_status")),
    psVar(ctx->subVar("psVar")),
    pvVar(ctx->subVar("pvVar")),
    ptVar(ctx->subVar("ptVar")),
    pNMaterials(ctx->subVar("pNMaterials")),
    positionName(""), particleIDs(""), archiveH(0),
    num_materials(0)
{ 

} 

//------------------------------------------------------------ 
ParticleFieldExtractor::~ParticleFieldExtractor(){} 

//------------------------------------------------------------- 

void ParticleFieldExtractor::add_type(string &type_list,
				      const TypeDescription *subtype)
{
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
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

bool ParticleFieldExtractor::setVars(DataArchive& archive)
{
  string command;

  names.clear();
  types.clear();
  archive.queryVariables(names, types);

  vector< double > times;
  vector< int > indices;
  archive.queryTimesteps( indices, times );
  GridP grid = archive.queryGrid(times[0]);
  LevelP level = grid->getLevel( 0 );
  Patch* r = *(level->patchesBegin());

  //string type_list("");
  //string name_list("");
  scalarVars.clear();
  vectorVars.clear();
  tensorVars.clear();
  pointVars.clear();
  particleIDVar = VarInfo();
  
  //  string ptNames;
  
  // reset the vars
  psVar.set("");
  pvVar.set("");
  ptVar.set("");

  
  // get all of the NC and Particle Variables
  const TypeDescription *td;
  bool found = false;
  for( int i = 0; i < (int)names.size(); i++ ){
    td = types[i];
    if(td->getType() ==  TypeDescription::ParticleVariable){
      const TypeDescription* subtype = td->getSubType();
      ConsecutiveRangeSet matls = archive.queryMaterials(names[i], r,
							 times[0]);
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
      case TypeDescription::int_type:
        scalarVars.push_back(VarInfo(names[i], matls));
	found = true;
	break;
      case  TypeDescription::Vector:
        vectorVars.push_back(VarInfo(names[i], matls));
	found = true;
	break;
      case  TypeDescription::Matrix3:
	tensorVars.push_back(VarInfo(names[i], matls));
	found = true;
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
  
  // get the number of materials for the NC & particle Variables
  num_materials = archive.queryNumMaterials(r, times[0]);
  cerr << "Number of Materials " << num_materials << endl;

  string visible;
  gui->eval(id + " isVisible", visible);
  if( visible == "1"){
     gui->execute(id + " destroyFrames");
     gui->execute(id + " build");
     gui->execute(id + " buildPMaterials " + to_string(num_materials));
     gui->execute(id + " buildVarList");    
  }
  return found;
}


void ParticleFieldExtractor::showVarsForMatls()
{
  ConsecutiveRangeSet onMaterials;
  for (int matl = 0; matl < num_materials; matl++) {
     string result;
     gui->eval(id + " isOn p" + to_string(matl), result);
     if ( result == "0")
	continue;
     onMaterials.addInOrder(matl);
  }

  bool needToUpdate = false;
  string spNames = getVarsForMaterials(scalarVars, onMaterials, needToUpdate);
  string vpNames = getVarsForMaterials(vectorVars, onMaterials, needToUpdate);
  string tpNames = getVarsForMaterials(tensorVars, onMaterials, needToUpdate);

  
  if (needToUpdate) {
    string visible;
    gui->eval(id + " isVisible", visible);
    if( visible == "1"){
      gui->execute(id + " clearVariables");
      gui->execute(id + " setParticleScalars " + spNames.c_str());
      gui->execute(id + " setParticleVectors " + vpNames.c_str());
      gui->execute(id + " setParticleTensors " + tpNames.c_str());
      gui->execute(id + " buildVarList");    
      gui->execute("update idletasks");
      reset_vars(); // ?? what is this for?  // It flushes the cache on varibles - Steve
    }
  }

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
}

string
ParticleFieldExtractor::getVarsForMaterials(list<VarInfo>& vars,
					    const ConsecutiveRangeSet& matls,
					    bool& needToUpdate)
{
  string names = "";
  list<VarInfo>::iterator iter;
  for (iter = vars.begin(); iter != vars.end(); iter++) {
     if (matls.intersected((*iter).matls).size() == matls.size()) {
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

void ParticleFieldExtractor::addGraphingVars(long64 particleID,
					     const list<VarInfo>& vars,
					     string type)
{
  list<VarInfo>::const_iterator iter;
  int i = 0;
  for (iter = vars.begin(); iter != vars.end(); iter++, i++) {
     ostringstream call;
     call << id << " addGraphingVar " << particleID << " " << (*iter).name <<
	" { " << get_matl_from_particleID(particleID) << " } " << type <<
       //	" {" << (*iter).matls.expandedString() << "} " << type <<
	" " << i;
     gui->execute(call.str().c_str());
  }
}

void ParticleFieldExtractor::callback(long64 particleID)
{
  cerr<< "ParticleFieldExtractor::callback request data for index "<<
    particleID << ".\n";

  ostringstream call;
  call << id << " create_part_graph_window " << particleID;
  gui->execute(call.str().c_str());
  addGraphingVars(particleID, scalarVars, "scalar");
  addGraphingVars(particleID, vectorVars, "vector");
  addGraphingVars(particleID, tensorVars, "matrix3");
}		
		
// This may need be made faster in the future.  Right now we are just looping
// over all the particle ID's for each materials searching for a match.
// Since this should only be used in debugging experiments it doesn't need
// to be super speedy, just responsive.
int ParticleFieldExtractor::get_matl_from_particleID(long64 particleID) {
  DataArchive& archive = *((*(archiveH.get_rep()))());
  GridP grid = archive.queryGrid( time );
  LevelP level = grid->getLevel( 0 );
  // loop over all the materials
  
  for(Level::const_patchIterator patch = level->patchesBegin();
      patch != level->patchesEnd();
      patch++ )
    {
      for(int matl = 0; matl < num_materials; matl++) {
	ParticleVariable< long64 > pvi;
	archive.query(pvi, particleIDs, matl, *patch, time);
	ParticleSubset* pset = pvi.getParticleSubset();
	// check if we have an particles on this patch
	if(pset->numParticles() > 0){
	  // now loop over the ParticleVariables and find it
	  for(ParticleSubset::iterator iter = pset->begin();
	      iter != pset->end(); iter++) {
	    if (pvi[*iter] == particleID)
	      return matl;
	  }
	}
      }
    }
  // failed to find a matl
  return -1;
}
/*

void ParticleFieldExtractor::graph(string idx, string var)
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
      ostr << id << " graph " << idx+var<<" "<<var << " ";
      int j = 0;
      for( i = tpr->GetStartTime(); i <= tpr->GetEndTime();
	   i += tpr->GetIncrement())
	{
	  ostr << i << " " << values[j++] << " ";
	}
      gui->execute( ostr.str().c_str() );
    }
  }
}
*/
//----------------------------------------------------------------
void ParticleFieldExtractor::execute() 
{ 
  tcl_status.set("Calling ParticleFieldExtractor!"); 
  bool newarchive;
  in = (ArchiveIPort *) get_iport("Data Archive");
  psout = (ScalarParticlesOPort *) get_oport("Scalar Particles");
  pvout = (VectorParticlesOPort *) get_oport("Vector Particles");
  ptout = (TensorParticlesOPort *) get_oport("Tensor Particles");
  ArchiveHandle handle;
   if(!in->get(handle)){
     warning("ParticleFieldExtractor::execute() Didn't get a handle.");
     return;
   }
   
   DataArchive& archive = *((*(handle.get_rep()))());

   if ( handle.get_rep() != archiveH.get_rep() ) {
     // we have a different archive
     
     // empty the cache of stored variables
     material_data_list.clear();
     
     if (archiveH.get_rep()  == 0 ){
       string visible;
       gui->eval(id + " isVisible", visible);
       if( visible == "0" ){
	 gui->execute(id + " buildTopLevel");
       }

     }
     
     if( !setVars( archive )){
       warning("Cannot read any ParticleVariables, no action.");
       return;
      }

     archiveH = handle;
    }
   showVarsForMatls();
     
   ScalarParticles* sp = 0;
   VectorParticles* vp = 0;
   TensorParticles* tp = 0;

   // what time is it?
   times.clear();
   indices.clear();
   archive.queryTimesteps( indices, times );
   int idx = handle->timestep();
   time = times[idx];

   buildData( archive, time, sp, vp, tp );
   psout->send( sp );
   pvout->send( vp );
   ptout->send( tp );	  
   tcl_status.set("Done");
}


void 
ParticleFieldExtractor::buildData(DataArchive& archive, double time,
				  ScalarParticles*& sp,
				  VectorParticles*& vp,
				  TensorParticles*& tp)
{

  
  GridP grid = archive.queryGrid( time );
  LevelP level = grid->getLevel( 0 );
  
  PSet* pset = new PSet();
  pset->SetLevel( level );
  pset->SetCallbackClass( this );
  

  bool have_sp = false;
  bool have_vp = false;
  bool have_tp = false;
  bool have_ids = false;
  int scalar_type;
  for(int i = 0; i < (int)names.size() ; i++)
    if (names[i] == psVar.get())
      scalar_type = types[i]->getSubType()->getType();

  int max_workers = Max(Thread::numProcessors()/2, 2);
  Semaphore* sema = scinew Semaphore( "scalar extractor semahpore",
				      max_workers); 
  Mutex smutex("ScalarParticles Mutex");
  Mutex vmutex("VectorParticles Mutex");
  Mutex tmutex("TensorrParticles Mutex");
  Mutex imutex("ParticleIds Mutex");
//   WallClockTimer my_timer;
//   my_timer.start();
  double size = level->numPatches();
  int count = 0;
  // iterate over patches
  for(Level::const_patchIterator patch = level->patchesBegin();
      patch != level->patchesEnd(); patch++ ){
//     update_progress(count++/size, my_timer);
    sema->down();
    Thread *thrd =
      new Thread( scinew PFEThread( this, archive,
				    *patch,  sp, vp, tp, pset,
				    scalar_type, have_sp, have_vp,
				    have_tp, have_ids, sema,
				    &smutex, &vmutex, &tmutex, &imutex, gui),
		  "Particle Field Extractor Thread");
    thrd->detach();
//     PFEThread *thrd = scinew PFEThread( this, archive, *patch,
// 			     sp, vp, tp, pset,
//  			     scalar_type, have_sp, have_vp,
//  			     have_tp, have_ids, sema,
//  			     &smutex, &vmutex, &tmutex, &imutex, gui);

//     thrd->run(); 
  }
  sema->down( max_workers );
  if( sema )  delete sema;
//   timer.add( my_timer.time() );
//   my_timer.stop();
} 
void PFEThread::run(){     


  ParticleSubset* dest_subset = scinew ParticleSubset();
  ParticleVariable< long64 > ids( dest_subset );
  ParticleVariable< Vector > vectors(dest_subset);
  ParticleVariable< Point > positions(dest_subset);
  ParticleVariable< double > scalars(dest_subset);
  ParticleVariable< Matrix3 > tensors( dest_subset );

  ParticleVariable< Vector > pvv;
  ParticleVariable< Matrix3 > pvt;
  ParticleVariable< double > pvs;
  ParticleVariable< Point  > pvp;
  ParticleVariable< int > pvint;
  ParticleVariable< long64 > pvi;

  //int numMatls = 29;

  for(int matl = 0; matl < pfe->num_materials; matl++) {
    string result;
    ParticleSubset* source_subset;
    bool have_subset = false;

    gui->eval(pfe->id + " isOn p" + to_string(matl), result);
    if ( result == "0")
      continue;
    if (pfe->pvVar.get() != ""){
      have_vp = true;
      archive.query(pvv, pfe->pvVar.get(), matl, patch, pfe->time);	
      if( !have_subset){
	source_subset = pvv.getParticleSubset();
	have_subset = true;
      }
    }
    if( pfe->psVar.get() != ""){
      have_sp = true;
      switch (scalar_type) {
      case TypeDescription::double_type:
	archive.query(pvs, pfe->psVar.get(), matl, patch, pfe->time);
	if( !have_subset){
	  source_subset = pvs.getParticleSubset();
	  have_subset = true;
	}
	break;
      case TypeDescription::int_type:
	//cerr << "Getting data for ParticleVariable<int>\n";
	archive.query(pvint, pfe->psVar.get(), matl, patch, pfe->time);
	if( !have_subset){
	  source_subset = pvi.getParticleSubset();
	  have_subset = true;
	}
	//cerr << "Got data\n";
	break;
      }
    }
    if (pfe->ptVar.get() != ""){
      have_tp = true;
      archive.query(pvt, pfe->ptVar.get(), matl, patch, pfe->time);
      if( !have_subset){
	source_subset = pvt.getParticleSubset();
	have_subset = true;
      }
    }
    if(pfe->positionName != "")
      archive.query(pvp, pfe->positionName, matl, patch, pfe->time);

    if(pfe->particleIDs != ""){
      //cerr<<"paricleIDs = "<<pfe->particleIDs<<endl;
      have_ids = true;
      archive.query(pvi, pfe->particleIDs, matl, patch, pfe->time);
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
    // 	<<"  mat "<<matl<<endl;
    //     pvv.getSizeInfo(elems, totsize, mem_start );
    //     cerr<<"there are "<<elems<<" vector elements for patch "<<patchn
    // 	<<"  mat "<<matl<<endl;
    //     pvt.getSizeInfo(elems, totsize, mem_start );
    //     cerr<<"there are "<<elems<<" tensor elements for patch "<<patchn
    // 	<<"  mat "<<matl<<endl;
      
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
  pset->AddParticles( positions, ids, patch);
  imutex->unlock();
  if(have_sp) {
    smutex->lock();
    if( sp == 0 ){
      sp = scinew ScalarParticles();
      sp->Set( PSetHandle(pset) );
    }
    sp->AddVar( scalars );
    smutex->unlock();
  } else 
    sp = 0;
  if(have_vp) {
    vmutex->lock();
    if( vp == 0 ){
      vp = scinew VectorParticles();
      vp->Set( PSetHandle(pset));
    }
    vp->AddVar( vectors );
    vmutex->unlock();
  } else 
    vp = 0;

  if(have_tp){
    tmutex->lock();
    if( tp == 0 ){
      tp = scinew TensorParticles();
      tp->Set( PSetHandle(pset) );
    }
    tp->AddVar( tensors);
    tmutex->unlock();
  } else
    tp = 0;

  sema->up();
}


void ParticleFieldExtractor::tcl_command(GuiArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  if(args[1] == "graph") {
    string varname(args[2]);
    string particleID(args[3]);
    int num_mat;
    string_to_int(args[4], num_mat);
    cerr << "Extracting " << num_mat << " materals:";
    vector< string > mat_list;
    vector< string > type_list;
    for (int i = 5; i < 5+(num_mat*2); i++) {
      string mat(args[i]);
      mat_list.push_back(mat);
      i++;
      string type(args[i]);
      type_list.push_back(type);
    }
    cerr << endl;
    cerr << "Graphing " << varname << " with materials: " << vector_to_string(mat_list) << endl;
    graph(varname,mat_list,type_list,particleID);
  }
  else {
    Module::tcl_command(args, userdata);
  }

}

void ParticleFieldExtractor::graph(string varname, vector<string> mat_list,
				   vector<string> type_list, string particleID)
{

  /* void DataArchive::query(std::vector<T>& values, const std::string& name,
    	    	    	int matlIndex, long particleID,
			double startTime, double endTime);
  */
  // clear the current contents of the ticles's material data list
  gui->execute(id + " reset_var_val");

  // determine type
  const TypeDescription *td;
  for(int i = 0; i < (int)names.size() ; i++)
    if (names[i] == varname)
      td = types[i];
  
  DataArchive& archive = *((*(this->archiveH.get_rep()))());
  vector< int > indices;
  times.clear();
  archive.queryTimesteps( indices, times );
  gui->execute(id + " setTime_list " + vector_to_string(indices).c_str());

  string name_list("");
  long64 partID = atoll(particleID.c_str());
  cout << "partID = "<<partID<<endl;
  cerr << "mat_list.size() = "<<mat_list.size()<<endl;
  for(int m = 0; m < (int)mat_list.size(); m++) {
    cerr << "mat_list["<<m<<"] = "<<mat_list[m]<<endl;
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
	cerr << "querying data archive for "<<varname<<" with matl="<<matl<<", particleID="<<partID<<", from time "<<times[0]<<" to time "<<times[times.size()-1]<<endl;
	try {
	  archive.query(values, varname, matl, partID, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  error("Particle Variable "+particleID+" not found.\n");
	  return;
	}
	cerr << "Received data.  Size of data = " << values.size() << endl;
	cache_value(particleID+" "+varname+" "+mat_list[i],values,data);
      } else {
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::int_type:
    cerr << "Graphing a variable of type double\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      // query the value
      if (!is_cached(particleID+" "+varname+" "+mat_list[i],data)) {
	// query the value and then cache it
	vector< int > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive.query(values, varname, matl, partID, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  error("Particle Variable "+particleID+" not found.\n");
	  return;
	}
	cerr << "Received data.  Size of data = " << values.size() << endl;
	cache_value(particleID+" "+varname+" "+mat_list[i],values,data);
      } else {
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Vector:
    cerr << "Graphing a variable of type Vector\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(particleID+" "+varname+" "+mat_list[i]+" "+type_list[i],
		     data)) {
	// query the value
	vector< Vector > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive.query(values, varname, matl, partID, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  error("Particle Variable "+particleID+" not found.\n");
	  return;
	}
	cerr << "Received data.  Size of data = " << values.size() << endl;
	data = vector_to_string(values,type_list[i]);
	cache_value(particleID+" "+varname+" "+mat_list[i],values);
      } else {
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";      
    }
    break;
  case TypeDescription::Matrix3:
    cerr << "Graphing a variable of type Matrix3\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(particleID+" "+varname+" "+mat_list[i]+" "+type_list[i],
		     data)) {
	// query the value
	vector< Matrix3 > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive.query(values, varname, matl, partID, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  error("Particle Variable "+particleID+" not found.\n");
	  return;
	}
	cerr << "Received data.  Size of data = " << values.size() << endl;
	data = vector_to_string(values,type_list[i]);
	cache_value(particleID+" "+varname+" "+mat_list[i],values);
      } else {
	cerr << "Cache hit\n";
      }
      gui->execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";      
    }
    break;
  default:
    cerr<<"Unknown var type\n";
  }// else { Tensor,Other}
  cerr << "callig graph_data with \"particleID="<<particleID<<" varname="<<varname<<" name_list="<<name_list<<endl;
  gui->execute(id+" graph_data "+particleID.c_str()+" "+varname.c_str()+" "+
	       name_list.c_str());

}

string ParticleFieldExtractor::vector_to_string(vector< int > data) {
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string ParticleFieldExtractor::vector_to_string(vector< string > data) {
  string result;
  for(int i = 0; i < (int)data.size(); i++) {
      result+= (data[i] + " ");
    }
  return result;
}

string ParticleFieldExtractor::vector_to_string(vector< double > data) {
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string ParticleFieldExtractor::vector_to_string(vector< Vector > data, string type) {
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

string ParticleFieldExtractor::vector_to_string(vector< Matrix3 > data, string type) {
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

bool ParticleFieldExtractor::is_cached(string name, string& data) {
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

void ParticleFieldExtractor::cache_value(string where, vector<double>& values,
				 string &data) {
  data = vector_to_string(values);
  material_data_list[where] = data;
}

void ParticleFieldExtractor::cache_value(string where, vector<int>& values,
				 string &data) {
  data = vector_to_string(values);
  material_data_list[where] = data;
}

void ParticleFieldExtractor::cache_value(string where, vector<Vector>& values)
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

void ParticleFieldExtractor::cache_value(string where, vector<Matrix3>& values)
{
  string data = vector_to_string(values,"Determinant");
  material_data_list[where+" Determinant"] = data;
  data = vector_to_string(values,"Trace");
  material_data_list[where+" Trace"] = data;
  data = vector_to_string(values,"Norm");
  material_data_list[where+" Norm"] = data;
}

} // End namespace Uintah
 
