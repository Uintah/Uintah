//static char *id="@(#) $Id$";

/****************************************
CLASS
    VizControl

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

#include <Uintah/Datatypes/Particles/MPVizParticleSet.h>
#include <Uintah/Datatypes/Particles/MPParticleGridReader.h>
#include <Uintah/Datatypes/Particles/MFMPParticleGridReader.h>
#include "VizControl.h"
#include <SCICore/Containers/String.h>
#include <iostream> 
#include <sstream>
using std::ostringstream;
#include <stdio.h>
#include <string.h>

namespace Uintah {
namespace Modules {

using namespace SCICore::Containers;

extern "C" PSECore::Dataflow::Module* make_VizControl( const clString& id ) {
  return new VizControl( id ); 
}

//--------------------------------------------------------------- 
VizControl::VizControl(const clString& id) 
  : Module("VizControl", id, Filter),
    tcl_status("tcl_status", id, this), gsVar("gsVar", id, this),
    gvVar("gvVar", id, this), psVar("psVar", id, this),
    pvVar("pvVar", id, this), pName("pName", id, this),
    gName("gName", id, this)
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=new ParticleGridReaderIPort(this, "ParticleGridReader",
				    ParticleGridReaderIPort::Atomic);
  sfout=new ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
  vfout=new VectorFieldOPort(this, "VectorField", VectorFieldIPort::Atomic);
  psout=new ParticleSetOPort(this, "ParticleSet", ParticleSetIPort::Atomic);

  // Add them to the Module
  add_iport(in);
  add_oport(sfout);
  add_oport(vfout);
  add_oport(psout);
  //add_oport(pseout);

} 

//------------------------------------------------------------ 
VizControl::~VizControl(){} 

//------------------------------------------------------------- 
void VizControl::tcl_command( TCLArgs& args, void* userdata)
{

  int i,j;
  clString result;
  std::cerr<<"tcl_command arg: "<< args[1]<<endl;
  
  if( args[1] == "getGridNames"){
    if(args.count() != 2 ) {
      args.error("VizControl--getGridNames");
      return;
    }
    if(pgrh.get_rep() == 0){
      return;
    }
    if( pgrh->GetNGrids() == 0 )
      result = "";
    else
      result = (pgrh->GetGrid( 0 ))->getName();
    
    for(i = 1; i < pgrh->GetNGrids(); i++){
      result += clString(" " + (pgrh->GetGrid( i ))->getName());
    }
    std::cerr<<"     result = "<< result<<endl;
    args.result( result );
  } else if( args[1] == "getParticleSetNames"){
    if(args.count() != 2 ) {
      cerr<< "#args = "<< args.count() << endl;
      args.error("VizControl--getParticleSetNames");
      return;
    }

    cerr << "I made it here \n";
    if(pgrh.get_rep() == 0){
      args.result("");
      return;
    }
    if( MPParticleGridReader* mppgr =
	dynamic_cast<MPParticleGridReader*> (pgrh.get_rep())){
      result = "";
      for(i = 0; i < mppgr->GetNParticleSets(); i++){
	ParticleSetHandle  psh = mppgr->GetParticleSet( i );
	if(MPVizParticleSet* mpvps =
	   dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	  result += mpvps->getName();
	}
	if(i < mppgr->GetNParticleSets()-1) result += clString(" ");
      }
      std::cerr<<"     result = "<< result<<endl;
      args.result( result );
    } else if ( MFMPParticleGridReader* mppgr =
	dynamic_cast<MFMPParticleGridReader*> (pgrh.get_rep())){
      result = "";
      for(i = 0; i < mppgr->GetNParticleSets(); i++){
	ParticleSetHandle  psh = mppgr->GetParticleSet( i );
	if(MPVizParticleSet* mpvps =
	   dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	  result += mpvps->getName();
	}
	if(i < mppgr->GetNParticleSets()-1) result += clString(" ");
      }
      std::cerr<<"     result = "<< result<<endl;
      args.result( result );
    } else {
      args.error("VizControl--getParticleSetName: Wrong subclass");
      return;
    }
  } else if( args[1] == "getParticleSetScalarNames"){
    if(args.count() != 3 ) {
      args.error("VizControl--getParticleSetScalarNames");
      return;
    }
    
    if(pgrh.get_rep() == 0){
      args.result("");
      return;
    }
    clString name = args[2];
    if( MPParticleGridReader* pgr =
	dynamic_cast<MPParticleGridReader *> (pgrh.get_rep()) ){
      Array1<clString> vars;
      for(i = 0; i < pgr->GetNParticleSets(); i++){
	ParticleSetHandle  psh = pgr->GetParticleSet( i );
	if(MPVizParticleSet* ps =
	   dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	  if(ps->getName() == name){
	    ps->list_scalars( vars );
	    if ( vars.size() > 0)
	      result = vars[0];
	    else
	      result = "";
	    for( j = 1; j < vars.size(); j++){
	      result += clString( " " + vars[j]);
	    }
	    std::cerr<<"     result = "<<result<<endl;
	    args.result( result );
	    break;
	  }
	}
      }
      if( i == pgr->GetNParticleSets() )
	args.error("VizControl--getParticleSetScalarNames--varname does not exist");
    } else if( MFMPParticleGridReader* pgr =
	dynamic_cast<MFMPParticleGridReader *> (pgrh.get_rep()) ){
      Array1<clString> vars;
      for(i = 0; i < pgr->GetNParticleSets(); i++){
	ParticleSetHandle  psh = pgr->GetParticleSet( i );
	if(MPVizParticleSet* ps =
	   dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	  if(ps->getName() == name){
	    ps->list_scalars( vars );
	    if ( vars.size() > 0)
	      result = vars[0];
	    else
	      result = "";
	    for( j = 1; j < vars.size(); j++){
	      result += clString( " " + vars[j]);
	    }
	    std::cerr<<"     result = "<<result<<endl;
	    args.result( result );
	    break;
	  }
	}
      }
      if( i == pgr->GetNParticleSets() )
	args.error("VizControl--getParticleSetScalarNames--varname does not exist");
    }
  } else if( args[1] == "getParticleSetVectorNames"){
    if(args.count() != 3 ) {
      args.error("VizControl--getParticleSetVectorNames");
      return;
    }
    if(pgrh.get_rep() == 0){
      args.result("");
      return;
    }
    clString name = args[2];
    if( MPParticleGridReader* pgr = 
	dynamic_cast<MPParticleGridReader *> (pgrh.get_rep()) ){
      Array1<clString> vars;
      for(i = 0; i < pgr->GetNParticleSets(); i++){
	ParticleSetHandle  psh = pgr->GetParticleSet( i );
	if(MPVizParticleSet* ps =
	   dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	  if(ps->getName() == name){
	    ps->list_vectors( vars );
	    result = "";
	    if ( vars.size() > 1)
	      result = vars[1];
	    else
	    for( j = 2; j < vars.size(); j++){
	      result += clString( " " + vars[j]);
	    }
	    std::cerr<<"     result = "<<result<<endl;
	    args.result( result );
	  }
	}
	if( i == pgr->GetNParticleSets() )
	  args.error("VizControl--getParticleSetScalarNames: varname does not exist");
      }
    }  else if( MFMPParticleGridReader* pgr = 
		dynamic_cast<MFMPParticleGridReader *> (pgrh.get_rep()) ){
      Array1<clString> vars;
      for(i = 0; i < pgr->GetNParticleSets(); i++){
	ParticleSetHandle  psh = pgr->GetParticleSet( i );
	if(MPVizParticleSet* ps =
	   dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	  if(ps->getName() == name){
	    ps->list_vectors( vars );
	    result = "";
	    if ( vars.size() > 1)
	      result = vars[1];
	    for( j = 2; j < vars.size(); j++){
	      result += clString( " " + vars[j]);
	    }
	    std::cerr<<"     result = "<<result<<endl;
	    args.result( result );
	  }
	}
	if( i == pgr->GetNParticleSets() )
	  args.error("VizControl--getParticleSetScalarNames: varname does not exist");
      }
    }  

  } else if( args[1] == "getGridScalarNames"){
    if(args.count() != 3 ) {
      args.error("VizControl--getGridScalarNames");
      return;
    }
    if(pgrh.get_rep() == 0){
      return;
    }
    clString name = args[2];
    if( MPParticleGridReader* pgr =
	dynamic_cast<MPParticleGridReader*> (pgrh.get_rep()) ){
      Array1<clString> vars;

      for(i = 0; i < pgr->GetNGrids(); i++){
	if(pgr->GetGrid( i )->getName() == name){
	  pgr->GetGrid( i )->getScalarNames( vars );
	  result = "";
	  if ( vars.size() > 0)
	    result = vars[0];
	  for( j = 1; j < vars.size(); j++){
	    result += clString( " " + vars[j]);

	  }

	  std::cerr<<"     result = "<<result<<endl;

	  args.result( result );
	}
	if( i == pgr->GetNGrids() )
	  args.error("VizControl--getGridScalarNames: varname does not exist");
      }
    } else if( MFMPParticleGridReader* pgr =
	dynamic_cast<MFMPParticleGridReader*> (pgrh.get_rep()) ){
      Array1<clString> vars;

      for(i = 0; i < pgr->GetNGrids(); i++){
	if(pgr->GetGrid( i )->getName() == name){
	  pgr->GetGrid( i )->getScalarNames( vars );
	  result = "";
	  if ( vars.size() > 0)
	    result = vars[0];
	  for( j = 1; j < vars.size(); j++){
	    result += clString( " " + vars[j]);

	  }

	  std::cerr<<"     result = "<<result<<endl;

	  args.result( result );
	}
	if( i == pgr->GetNGrids() )
	  args.error("VizControl--getGridScalarNames: varname does not exist");
      }
    }
  } else if( args[1] == "getGridVectorNames"){
    if(args.count() != 3 ) {
      args.error("VizControl--getGridVectorNames");
      return;
    }
    if(pgrh.get_rep() == 0){
      return;
    }
    clString name = args[2];
    if( MPParticleGridReader* pgr = 
	dynamic_cast<MPParticleGridReader*> (pgrh.get_rep()) ){
      Array1<clString> vars;
      for(i = 0; i < pgr->GetNGrids(); i++){
	if(pgr->GetGrid( i )->getName() == name){
	  pgr->GetGrid( i )->getVectorNames( vars );
	  result = "";
	  if ( vars.size() > 0)
	    result = vars[0];
	  for( j = 1; j < vars.size(); j++){
	    result += clString( " " + vars[j]);
	  }
	  std::cerr<<"     result = "<<result<<endl;
	  args.result( result );
	}
	if( i == pgr->GetNGrids() )
	  args.error("VizControl--getGridVectorNames: varname does not exist");
      }
    } else if( MFMPParticleGridReader* pgr = 
	dynamic_cast<MFMPParticleGridReader*> (pgrh.get_rep()) ){
      Array1<clString> vars;
      for(i = 0; i < pgr->GetNGrids(); i++){
	if(pgr->GetGrid( i )->getName() == name){
	  pgr->GetGrid( i )->getVectorNames( vars );
	  result = "";
	  if ( vars.size() > 0)
	    result = vars[0];
	  for( j = 1; j < vars.size(); j++){
	    result += clString( " " + vars[j]);
	  }
	  std::cerr<<"     result = "<<result<<endl;
	  args.result( result );
	}
	if( i == pgr->GetNGrids() )
	  args.error("VizControl--getGridVectorNames: varname does not exist");
      }
    }  
  } else if (args[1] == "graph" ) {
    clString var = args[3];
    clString idx = args[2];
    graph(idx, var);
  } else {
    Module::tcl_command(args, userdata);
  }
}


//////////
// If receiving data for the first time, this function selects
// the first variable in each catagory.  If a previously save script
// has been loaded, the first data to be recieved will appear new even
// though the data being viewed is probably the same as that being 
// viewed at some previous time.  Several "if" statements are used
// to determine if the previously selected variables can still be 
// used.
//
void VizControl::setVars(ParticleGridReaderHandle reader)
{
  clString command;
  if(MPParticleGridReader *r =
     dynamic_cast<MPParticleGridReader*> (reader.get_rep())){

    TCL::execute(id + " destroyFrames");
    int i;
    Array1< clString > Vars;
    for(i = 0; i < r->GetNGrids(); i++ ){
      TCL::execute(id + " addName grid " + r->GetGrid(i)->getName() );
    }
    for(i = 0; i < r->GetNParticleSets(); i++ ){
      ParticleSetHandle psh = r->GetParticleSet(i);
      if(MPVizParticleSet * ps = 
	 dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	command = id + " addName particleSet " + ps->getName();
	TCL::execute(command);
      }
    }
    
    reset_vars();
    NOT_FINISHED("VizControl::setVars()");
  }
}

//////////
// Check to see that this data has the same variables as the last
// set of data.  If they are different then call setVars, because
// previous selections do not apply.
void VizControl::checkVars(ParticleGridReaderHandle reader)
{
  if(MPParticleGridReader *r =
     dynamic_cast<MPParticleGridReader*> (reader.get_rep())){
    if (MPParticleGridReader *pgr =
	dynamic_cast<MPParticleGridReader*>  (pgrh.get_rep())){
      int i,j;

      int rGrids = r->GetNGrids();
      int pgrGrids = pgr->GetNGrids();
      if ( rGrids != pgrGrids){
	setVars( reader );
	return;
      }
    
      int rParticleSets = r->GetNParticleSets();
      int pgrParticleSets = pgr->GetNParticleSets();
      if(rParticleSets != pgrParticleSets){
	setVars(reader);
	return;
      }
      
      for( i = 0; i < rGrids; i++){
	if( r->GetGrid(i)->getName() !=
	    pgr->GetGrid(i)->getName() ){
	  setVars(reader);
	  return;
	}
      }

      for( i = 0; i < rParticleSets; i++){
	ParticleSetHandle psh = r->GetParticleSet(i);
       	ParticleSetHandle pgh = pgr->GetParticleSet(i);
	if(MPVizParticleSet * ps =
	   dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
	  if(MPVizParticleSet * psr =
	     dynamic_cast<MPVizParticleSet*> (pgh.get_rep())){
	    if( ps->getName() !=  psr->getName() ){
	      setVars(reader);
	      return;
	    }
	  }
	}
      }
      Array1< clString > rVars;
      Array1< clString > pgrVars;
      for( i = 0; i < rGrids; i++){
	r->GetGrid(i)->getScalarNames(rVars);
	pgr->GetGrid(i)->getScalarNames(pgrVars);
	if( rVars.size() == pgrVars.size()){
	  for(j = 0; j < rVars.size(); j++)
	    if( rVars[i] != pgrVars[i] ){
	      setVars(reader);
	      return;
	    }
	} else {
	  setVars(reader);
	  return;
	}
      }
      
      for( i = 0; i < rGrids; i++){
	r->GetGrid(i)->getVectorNames(rVars);
	pgr->GetGrid(i)->getVectorNames(pgrVars);
	if( rVars.size() == pgrVars.size()){
	  for(j = 0; j < rVars.size(); j++)
	    if( rVars[i] != pgrVars[i] ){
	      setVars(reader);
	      return;
	    }
	} else {
	  setVars(reader);
	  return;
	}
      }
      
      for( i = 0; i < rParticleSets; i++){
	r->GetParticleSet(i)->list_scalars(rVars);
	pgr->GetParticleSet(i)->list_scalars(pgrVars);
	if( rVars.size() == pgrVars.size()){
	  for(j = 0; j < rVars.size(); j++)
	    if( rVars[i] != pgrVars[i] ){
	      setVars(reader);
	      return;
	    }
	} else {
	  setVars(reader);
	  return;
	}
      }

      for( i = 0; i < rParticleSets; i++){
	r->GetParticleSet(i)->list_vectors(rVars);
	pgr->GetParticleSet(i)->list_vectors(pgrVars);
	if( rVars.size() == pgrVars.size()){
	  for(j = 0; j < rVars.size(); j++)
	    if( rVars[i] != pgrVars[i] ){
	      setVars(reader);
	      return;
	    }
	} else {
	  setVars(reader);
	  return;
	}
      }
    }
  }
}

void VizControl::callback( int index)
{
  cerr<< "VizControl::callback request data for index "<<
    index << ".\n";

  clString idx = to_string(index);
  TCL::execute( id + " infoFrame " + idx);
  TCL::execute( id + " infoAdd " + idx + " " + "0" + 
		" Info for particle " + idx + " in material "
		+ pName.get() + ".\n");

  int i;
  Array1< clString> vars;
  if( MPParticleGridReader* tpr =
      dynamic_cast<MPParticleGridReader*> (pgrh.get_rep())){
    ParticleSetHandle psh = tpr->GetParticleSet(pName.get());
    if(MPVizParticleSet *ps = dynamic_cast<MPVizParticleSet*> (psh.get_rep())){
      ps->list_scalars( vars );
      int timestep= 0;
      for(i = 0; i < vars.size(); i++){
	int sid = ps->find_scalar( vars[i] );
	double scale = ps->getScalar(timestep, sid, index);
	TCL::execute( id + " infoAdd " + idx + " "
		      + to_string(tpr->GetNTimesteps())
		      + " " +  vars[i] + " = " + to_string( scale ));
      }
      
      vars.remove_all();
      ps->list_vectors( vars );
      for(i = 0; i < vars.size(); i++) {
	int vid = ps->find_vector( vars[i] );
	Vector v = ps->getVector(timestep, vid, index);
	TCL::execute( id + " infoAdd " + idx + " " + "0" + " " +
		      vars[i] + " = (" + to_string( v.x() ) +
		      ", " + to_string(v.y()) + ", " +
		      to_string(v.z()) + ")" );
      }
      
    }
  }
}
void VizControl::graph(clString idx, clString var)
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

//----------------------------------------------------------------
void VizControl::execute() 
{ 
   tcl_status.set("Calling VizControl!"); 

   ParticleGridReaderHandle handle;
   if(!in->get(handle)){
     std::cerr<<"Didn't get a handle\n";
     return;
   }
  
   if ( handle.get_rep() != pgrh.get_rep() ) {
       
     if (pgrh.get_rep()  == 0 ){
       setVars( handle );
       
     } else {
       checkVars( handle );
     }
     pgrh = handle;
   }       
     
   if ( gsVar.get() != ""){
     ScalarFieldHandle sfh = pgrh->GetGrid( gName.get() )->
       getScalarField( gsVar.get() );
     if ( sfh.get_rep() ){
       sfout->send(sfh);
       std::cerr<<"Scalarfield name is "<<gsVar.get()<<endl;
     }
   }
   if( gvVar.get() != ""){
     VectorFieldHandle vfh = pgrh->GetGrid( gName.get() )->
       getVectorField( gvVar.get() );
     if( vfh.get_rep() ){
       vfout->send(vfh);
       std::cerr<<"Vectorfield name is "<<gvVar.get()<<endl;
     }
   }

   if(psVar.get() != "" || pvVar.get() != "" ){
     ParticleSetHandle psh = pgrh->GetParticleSet( pName.get() );
     if( MPVizParticleSet *mpvps =
	 dynamic_cast <MPVizParticleSet *> (psh.get_rep())){
       mpvps->SetScalarId( psVar.get());
       mpvps->SetVectorId( pvVar.get());
       mpvps->SetCallback( (void *) this);
     }
     if( psh.get_rep() ){
	 std::cerr<<"ParticleSet vars are " <<  psVar.get()<<" and  "<<
	   pvVar.get()<<endl;
       psout->send(psh);
     }
   }
} 
//--------------------------------------------------------------- 
} // end namespace CFD
} // end namespace SCI
  
