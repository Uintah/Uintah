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
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticles.h>
#include <Packages/Uintah/Core/Datatypes/ScalarParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticles.h>
#include <Packages/Uintah/Core/Datatypes/VectorParticlesPort.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticles.h>
#include <Packages/Uintah/Core/Datatypes/TensorParticlesPort.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <iostream> 
#include <sstream>
#include <string>

namespace Uintah {

using std::cerr;
using std::endl;
using std::vector;
using std::string;

using namespace SCIRun;

extern "C" Module* make_ParticleFieldExtractor( const clString& id ) {
  return scinew ParticleFieldExtractor( id ); 
}

//--------------------------------------------------------------- 
ParticleFieldExtractor::ParticleFieldExtractor(const clString& id) 
  : Module("ParticleFieldExtractor", id, Filter),
    tcl_status("tcl_status", id, this),
    psVar("psVar", id, this),
    pvVar("pvVar", id, this),
    ptVar("ptVar", id, this),
    pNMaterials("pNMaterials", id, this),
    archive(0), positionName(""), particleIDs("")
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=scinew ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  psout=scinew ScalarParticlesOPort(this, "ScalarParticles",
				 ScalarParticlesIPort::Atomic);
  pvout=scinew VectorParticlesOPort(this, "VectorParticles",
				 VectorParticlesIPort::Atomic);
  ptout=scinew TensorParticlesOPort(this, "TensorParticles",
				 TensorParticlesIPort::Atomic);

  // Add them to the Module
  add_iport(in);
  add_oport(psout);
  add_oport(pvout);
  add_oport(ptout);
  //add_oport(pseout);

} 

//------------------------------------------------------------ 
ParticleFieldExtractor::~ParticleFieldExtractor(){} 

//------------------------------------------------------------- 

void ParticleFieldExtractor::add_type(string &type_list,
				      const TypeDescription *subtype)
{
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
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

void ParticleFieldExtractor::setVars(ArchiveHandle ar)
{
  string command;
  DataArchive& archive = *((*(ar.get_rep()))());

  names.clear();
  types.clear();
  archive.queryVariables(names, types);

  vector< double > times;
  vector< int > indices;
  archive.queryTimesteps( indices, times );

  string spNames("");
  string vpNames("");
  string tpNames("");
  string type_list("");
  string name_list("");
  //  string ptNames;
  
  // reset the vars
  psVar.set("");
  pvVar.set("");
  ptVar.set("");

  
  // get all of the NC and Particle Variables
  const TypeDescription *td;
  for( int i = 0; i < (int)names.size(); i++ ){
    td = types[i];
    if(td->getType() ==  TypeDescription::ParticleVariable){
      const TypeDescription* subtype = td->getSubType();
      switch ( subtype->getType() ) {
      case TypeDescription::double_type:
	spNames += " ";
	spNames += names[i];
	if( psVar.get() == "" ){ psVar.set( names[i].c_str() ); }
	cerr << "Added scalar particle: " << names[i] << '\n';
	add_type(type_list,types[i]->getSubType());
	name_list += " ";
	name_list += names[i];
	break;
      case  TypeDescription::Vector:
	vpNames += " ";
	vpNames += names[i];
	if( pvVar.get() == "" ){ pvVar.set( names[i].c_str() ); }
	cerr << "Added vector particle: " << names[i] << '\n';
	add_type(type_list,types[i]->getSubType());
	name_list += " ";
	name_list += names[i];
	break;
      case  TypeDescription::Matrix3:
	tpNames += " ";
	tpNames += names[i];
	if( ptVar.get() == "" ){ ptVar.set( names[i].c_str() ); }
	cerr << "Added tensor particle: " << names[i] << '\n';
	add_type(type_list,types[i]->getSubType());
	name_list += " ";
	name_list += names[i];
	break;
      case  TypeDescription::Point:
	positionName = names[i];
	break;
      case TypeDescription::long_type:
	if( names[i] == "p.particleID" )
	  particleIDs = names[i];
	else
	  particleIDs = "";
      default:
	cerr<<"Unknown particle type\n";
      }// else { Tensor,Other}
    }
  }
  
  // get the number of materials for the NC & particle Variables
  GridP grid = archive.queryGrid(times[0]);
  LevelP level = grid->getLevel( 0 );
  Patch* r = *(level->patchesBegin());
  int numpsMatls = archive.queryNumMaterials(psVar.get()(), r, times[0]);
  
  clString visible;
  TCL::eval(id + " isVisible", visible);
  if( visible == "1"){
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");
    
    TCL::execute(id + " buildPMaterials " + to_string(numpsMatls));
    pNMaterials.set(numpsMatls);
    
    TCL::execute(id + " setParticleScalars " + spNames.c_str());
    TCL::execute(id + " setParticleVectors " + vpNames.c_str());
    TCL::execute(id + " setParticleTensors " + tpNames.c_str());
    TCL::execute(id + " buildVarList");
    TCL::execute(id + " setVar_list " + name_list.c_str());
    TCL::execute(id + " setType_list " + type_list.c_str());  
    TCL::execute("update idletasks");
    reset_vars();
  }
}



void ParticleFieldExtractor::callback(long particleID)
{
  cerr<< "ParticleFieldExtractor::callback request data for index "<<
    particleID << ".\n";

  ostringstream call;
  call << id << " create_part_graph_window " << particleID;
  TCL::execute(call.str().c_str());


//   if( this->archive.get_rep() == 0 ) return;

//   DataArchive& archive = *((*(this->archive.get_rep()))());
//   vector< double > times;
//   vector< int > indices;
//   archive.queryTimesteps( indices, times );
//   double time;
//   GridP grid = archive.queryGrid( time );
//   LevelP level = grid->getLevel( 0 );

//   int matl = 0;
//   double starttime = times[0];
//   double endtime = times[times.size() -1];

//   vector<double> sparts;
  
//   archive.query(sparts, psVar.get()(), 0, index, starttime, endtime);

//   vector<double>::iterator iter = sparts.begin();
//   for(; iter < sparts.end(); iter++){
//     cerr<<sparts[*iter]<<" ";
//   }
//   cerr<<endl;
 
  
//   clString idx = to_string(index);
//   TCL::execute( id + " infoFrame " + idx);
//   TCL::execute( id + " infoAdd " + idx + " " + "0" + 
// 		" Info for particle " + idx + " in material "
// 		+ pName.get() + ".\n");
  /*
  TCL::execute( id + " infoAdd " + idx + " "
		+ to_string(tpr->GetNTimesteps())
		+ " " +  vars[i] + " = " + to_string( scale ));
   
      
  Vector v = ps->getVector(timestep, vid, index);
  TCL::execute( id + " infoAdd " + idx + " " + "0" + " " +
		vars[i] + " = (" + to_string( v.x() ) +
		", " + to_string(v.y()) + ", " +
		to_string(v.z()) + ")" );
  */
}		
		
		
/*

void ParticleFieldExtractor::graph(clString idx, clString var)
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
      TCL::execute( ostr.str().c_str() );
    }
  }
}
*/
//----------------------------------------------------------------
void ParticleFieldExtractor::execute() 
{ 
  tcl_status.set("Calling ParticleFieldExtractor!"); 
  
  ArchiveHandle handle;
   if(!in->get(handle)){
     std::cerr<<"Didn't get a handle\n";
     return;
   }
   
   cerr << "Calling setVars\n";
   if ( handle.get_rep() != archive.get_rep() ) {
     
     if (archive.get_rep()  == 0 ){
       clString visible;
       TCL::eval(id + " isVisible", visible);
       if( visible == "0" ){
	 TCL::execute(id + " buildTopLevel");
       }
     }
     setVars( handle );
     archive = handle;
   }       
     
   cerr << "done with setVars\n";



   DataArchive& archive = *((*(handle.get_rep()))());
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
  pset->SetCallbackClass( this );
  
  
   
  ParticleSubset* dest_subset = scinew ParticleSubset();
  ParticleVariable< long > ids( dest_subset );
  ParticleVariable< Vector > vectors(dest_subset);
  ParticleVariable< Point > positions(dest_subset);
  ParticleVariable< double > scalars(dest_subset);
  ParticleVariable< Matrix3 > tensors( dest_subset );

  bool have_sp = false;
  bool have_vp = false;
  bool have_tp = false;
  bool have_ids = false;
  // iterate over patches
  for(Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); r++ ){
    ParticleVariable< Vector > pvv;
    ParticleVariable< Matrix3 > pvt;
    ParticleVariable< double > pvs;
    ParticleVariable< Point  > pvp;
    ParticleVariable< long > pvi;
     
    int numMatls = archive.queryNumMaterials(positionName, *r, time);
    //int numMatls = 29;
    for(int matl=0;matl<numMatls;matl++){
      clString result;
      eval(id + " isOn p" + to_string(matl), result);
      if ( result == "0")
	continue;
      if (pvVar.get() != ""){
	have_vp = true;
	archive.query(pvv, string(pvVar.get()()), matl, *r, time);
      }
      if( psVar.get() != ""){
	have_sp = true;
	archive.query(pvs, string(psVar.get()()), matl, *r, time);
      }
      if (ptVar.get() != ""){
	have_tp = true;
	archive.query(pvt, string(ptVar.get()()), matl, *r, time);
      }
      if(positionName != "")
	archive.query(pvp, positionName, matl, *r, time);

      if(particleIDs != ""){
	cerr<<"paricleIDs = "<<particleIDs<<endl;
	have_ids = true;
	archive.query(pvi, particleIDs, matl, *r, time);
      }
      
      ParticleSubset* source_subset = pvs.getParticleSubset();
      particleIndex dest = dest_subset->addParticles(source_subset->numParticles());
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
	  scalars[dest]=pvs[*iter];
	else
	  scalars[dest]=0;
	if(have_tp)
	  tensors[dest]=pvt[*iter];
	else
	  tensors[dest]=Matrix3(0.0);
	if(have_ids)
	  ids[dest] = pvi[*iter];
	else
	  ids[dest] = -1;
	
	positions[dest]=pvp[*iter];
      }
    }
    pset->AddParticles( positions, ids, *r);
    
    if(have_sp) {
      if( sp == 0 ){
	sp = scinew ScalarParticles();
	sp->Set( PSetHandle(pset) );
      }
      sp->AddVar( scalars );
    } else 
      sp = 0;
    if(have_vp) {
      if( vp == 0 ){
	vp = scinew VectorParticles();
	vp->Set( PSetHandle(pset));
      }
      vp->AddVar( vectors );
    } else 
      vp = 0;

    if(have_tp){
      if( tp == 0 ){
	tp = scinew TensorParticles();
	tp->Set( PSetHandle(pset) );
      }
      tp->AddVar( tensors);
    } else
      tp = 0;
  }
} 

void ParticleFieldExtractor::tcl_command(TCLArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  if(args[1] == "graph") {
    string varname(args[2]());
    string particleID(args[3]());
    int num_mat;
    args[4].get_int(num_mat);
    cerr << "Extracting " << num_mat << " materals:";
    vector< string > mat_list;
    vector< string > type_list;
    for (int i = 5; i < 5+(num_mat*2); i++) {
      string mat(args[i]());
      mat_list.push_back(mat);
      i++;
      string type(args[i]());
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
  TCL::execute(id + " reset_var_val");

  // determine type
  const TypeDescription *td;
  for(int i = 0; i < (int)names.size() ; i++)
    if (names[i] == varname)
      td = types[i];
  
  DataArchive& archive = *((*(this->archive.get_rep()))());
  vector< int > indices;
  times.clear();
  archive.queryTimesteps( indices, times );
  TCL::execute(id + " setTime_list " + vector_to_string(indices).c_str());

  string name_list("");
  long partID = atol(particleID.c_str());
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
	archive.query(values, varname, matl, partID, times[0], times[times.size()-1]);
	cerr << "Received data.  Size of data = " << values.size() << endl;
	cache_value(particleID+" "+varname+" "+mat_list[i],values,data);
      } else {
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
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
	archive.query(values, varname, matl, partID, times[0], times[times.size()-1]);
	cerr << "Received data.  Size of data = " << values.size() << endl;
	data = vector_to_string(values,type_list[i]);
	cache_value(particleID+" "+varname+" "+mat_list[i],values);
      } else {
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
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
	archive.query(values, varname, matl, partID, times[0], times[times.size()-1]);
	cerr << "Received data.  Size of data = " << values.size() << endl;
	data = vector_to_string(values,type_list[i]);
	cache_value(particleID+" "+varname+" "+mat_list[i],values);
      } else {
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";      
    }
    break;
  default:
    cerr<<"Unknown var type\n";
  }// else { Tensor,Other}
  TCL::execute(id+" graph_data "+particleID.c_str()+" "+varname.c_str()+" "+
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
 
