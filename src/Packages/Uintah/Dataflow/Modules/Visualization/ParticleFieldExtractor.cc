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

#include <SCICore/Util/NotFinished.h>
#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/TypeDescription.h>
#include <Uintah/Datatypes/ScalarParticles.h>
#include <Uintah/Datatypes/ScalarParticlesPort.h>
#include <Uintah/Datatypes/VectorParticles.h>
#include <Uintah/Datatypes/VectorParticlesPort.h>
#include <Uintah/Datatypes/TensorParticles.h>
#include <Uintah/Datatypes/TensorParticlesPort.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h>
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace Uintah {
namespace Modules {

using SCICore::Containers::to_string;
using namespace SCICore::TclInterface;
using namespace Uintah::Datatypes;
using PSECore::Datatypes::ScalarParticlesIPort;
using PSECore::Datatypes::ScalarParticlesOPort;
using PSECore::Datatypes::VectorParticlesIPort;
using PSECore::Datatypes::VectorParticlesOPort;
using PSECore::Datatypes::TensorParticlesIPort;
using PSECore::Datatypes::TensorParticlesOPort;

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
    archive(0), positionName("")
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

  void ParticleFieldExtractor::setVars(ArchiveHandle ar)
{
  string command;
  DataArchive& archive = *((*(ar.get_rep()))());

  vector< string > names;
  vector< const TypeDescription *> types;
  archive.queryVariables(names, types);

  vector< double > times;
  vector< int > indices;
  archive.queryTimesteps( indices, times );

  string spNames("");
  string vpNames("");
  string tpNames("");
  //  string ptNames;
  
  // reset the vars
  psVar.set("");
  pvVar.set("");
  ptVar.set("");

  // get all of the NC and Particle Variables
  const TypeDescription *td;
  for( int i = 0; i < names.size(); i++ ){
    td = types[i];
    if(td->getType() ==  TypeDescription::ParticleVariable){
       const TypeDescription* subtype = td->getSubType();
       switch ( subtype->getType() ) {
       case TypeDescription::double_type:
	 spNames += " ";
	 spNames += names[i];
	 if( psVar.get() == "" ){ psVar.set( names[i].c_str() ); }
	 cerr << "Added scalar particle: " << names[i] << '\n';
	 break;
       case  TypeDescription::Vector:
	 vpNames += " ";
	 vpNames += names[i];
	 if( pvVar.get() == "" ){ pvVar.set( names[i].c_str() ); }
	 cerr << "Added vector particle: " << names[i] << '\n';
	 break;
       case  TypeDescription::Matrix3:
	 tpNames += " ";
	 tpNames += names[i];
	 if( ptVar.get() == "" ){ ptVar.set( names[i].c_str() ); }
	 cerr << "Added tensor particle: " << names[i] << '\n';
	 break;
       case  TypeDescription::Point:
	positionName = names[i];
	break;
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

    TCL::execute("update idletasks");
    reset_vars();
  }
}



void ParticleFieldExtractor::callback( int index)
{
  cerr<< "ParticleFieldExtractor::callback request data for index "<<
    index << ".\n";
  /*
  clString idx = to_string(index);
  TCL::execute( id + " infoFrame " + idx);
  TCL::execute( id + " infoAdd " + idx + " " + "0" + 
		" Info for particle " + idx + " in material "
		+ pName.get() + ".\n");

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
   vector< double > times;
   vector< int > indices;
   archive.queryTimesteps( indices, times );
   int idx = handle->timestep();
   double time = times[idx];

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

   
  ParticleSubset* dest_subset = scinew ParticleSubset();
  ParticleVariable< Vector > vectors(dest_subset);
  ParticleVariable< Point > positions(dest_subset);
  ParticleVariable< double > scalars(dest_subset);
  ParticleVariable< Matrix3 > tensors( dest_subset );

  bool have_sp = false;
  bool have_vp = false;
  bool have_tp = false;
  
  // iterate over patches
  for(Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); r++ ){
    ParticleVariable< Vector > pvv;
    ParticleVariable< Matrix3 > pvt;
    ParticleVariable< double > pvs;
    ParticleVariable< Point  > pvp;
     
    int numMatls = archive.queryNumMaterials(positionName, *r, time);

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
      
      ParticleSubset* source_subset = pvs.getParticleSubset();
      particleIndex dest = dest_subset->addParticles(source_subset->numParticles());
      vectors.resync();
      positions.resync();
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
	positions[dest]=pvp[*iter];
	if(have_tp)
	  tensors[dest]=Matrix3(0.0);
	else
	  tensors[dest]=pvt[*iter];
      }
    }
  }
  if(have_sp) {
    sp =  scinew ScalarParticles( positions, scalars, this);
    sp->SetCallbackClass( this );
  } else 
    sp = 0;
  if(have_tp){
    tp =  scinew TensorParticles( positions, tensors, this);
    tp->SetCallbackClass( this );
  } else
    tp = 0;
  if(have_vp){
    vp = scinew VectorParticles( positions, vectors, this);
    vp->SetCallbackClass( this );
  } else 
    vp = 0;
} 

//--------------------------------------------------------------- 
} // end namespace Modules
} // end namespace Kurt
  
