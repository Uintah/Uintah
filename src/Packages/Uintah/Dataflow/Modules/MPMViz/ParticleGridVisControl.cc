//static char *id="@(#) $Id$";

/****************************************
CLASS
    ParticleGridVisControl

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

#include <Uintah/Datatypes/Particles/MPMaterial.h>
#include <Uintah/Datatypes/Particles/MPVizParticleSet.h>
#include "ParticleGridVisControl.h"
#include <Uintah/Datatypes/Particles/TecplotReader.h>

#include <SCICore/Util/NotFinished.h>
#include <SCICore/TclInterface/Histogram.h>

#include <SCICore/Malloc/Allocator.h>

#include <string.h>
#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <sstream>
using std::ostringstream;

namespace Uintah {
namespace Modules {

using namespace SCICore::Containers;

extern "C" PSECore::Dataflow::Module* make_ParticleGridVisControl( const clString& id ) {
  return scinew ParticleGridVisControl( id ); 
}

//--------------------------------------------------------------- 
ParticleGridVisControl::ParticleGridVisControl(const clString& id) 
  : Module("ParticleGridVisControl", id, Filter),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    vVar("vVar", id, this), psVar("psVar", id, this),
    pvVar("pvVar", id, this), sMaterial("sMaterial", id, this),
    vMaterial("vMaterial", id, this), pMaterial("pMaterial", id, this)
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=scinew ParticleGridReaderIPort(this, "ParticleGridReader",
				  ParticleGridReaderIPort::Atomic);
  sfout=scinew ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
  vfout=scinew VectorFieldOPort(this, "VectorField", VectorFieldIPort::Atomic);
  psout=scinew ParticleSetOPort(this, "ParticleSet", ParticleSetIPort::Atomic);
  //  pseout = scinew ParticleSetExtensionOPort(this, "ParticleSetExtension",
  //				 ParticleSetExtensionIPort::Atomic);
  // Add them to the Module
  add_iport(in);
  add_oport(sfout);
  add_oport(vfout);
  add_oport(psout);
  //  add_oport(pseout);

} 

//------------------------------------------------------------ 
ParticleGridVisControl::~ParticleGridVisControl(){} 


//-------------------------------------------------------------- 
void ParticleGridVisControl::tcl_command( TCLArgs& args, void* userdata)
{

  int i;
  clString result;
  MPMaterial *mat;
  
  static int counter = 0;
  
  if (args[1] == "getScalarNames") {
    if (args.count() != 2) {
      args.error("ParticleGridVisControl--getScalarNames");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    cerr<<"Before Tecplot test "<< counter++ << endl;
    if( TecplotReader *tpr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      cerr<<"After Tecplot test "<< counter -1<< endl;
      if ( tpr->getNMaterials() ){
	mat = tpr->getMaterial( 0 );
	Array1<clString> varnames;
	mat->getScalarNames( varnames );
	result = varnames[0];
	for( i = 1; i < varnames.size(); i++) {
	  result += clString( " " + varnames[i]);
	}
	args.result( result );
      }
    }
  } else if (args[1] == "getVectorNames") {
    if (args.count() != 2) {
      args.error("ParticleGridVisControl--getVectorNames");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    cerr<<"Before Tecplot test "<< counter++ << endl;
    if( TecplotReader *tpr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      cerr<<"After Tecplot test "<< counter -1<< endl;
      if ( tpr->getNMaterials() ){
	mat = tpr->getMaterial( 0 );
	Array1<clString> varnames;
	mat->getVectorNames( varnames );
	result = varnames[0];
	for( i = 1; i < varnames.size(); i++) {
	  result += clString( " " + varnames[i]);
	}
	args.result( result );
      }
    }
  } else if (args[1] == "getNMaterials" ) {
    if (args.count() != 2) {
      args.error("ParticleGridVisControl--getNMaterials");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    cerr<<"Before Tecplot test "<< counter++ << endl;
    if( TecplotReader *tpr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      cerr<<"After Tecplot test "<< counter -1<< endl;
      int n = tpr->getNMaterials();
      result = to_string( n );
      args.result( result );
    }
  } else if (args[1] == "hasParticles" ) {
    if (args.count() != 3) {
      args.error("ParticleGridVisControl--hasParticles needs an index.");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    cerr<<"Before Tecplot test "<< counter++ << endl;
    if( TecplotReader *tpr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      cerr<<"After Tecplot test "<< counter -1<< endl;
      int f = atoi ( args[2]() );
      mat = tpr->getMaterial( f - 1 );
      ParticleSetHandle psh = mat->getParticleSet();
      if( psh.get_rep() == 0 )
	args.result( clString( "0" ));
      else
	args.result( clString( "1" ));
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
void ParticleGridVisControl::setVars(ParticleGridReaderHandle reader)
{
  if( TecplotReader *tpr = dynamic_cast<TecplotReader*> (reader.get_rep())){
    int nMaterials = tpr->getNMaterials();
    bool reset = false;
    if (!(sMaterial.get() <= nMaterials &&
	  vMaterial.get() <= nMaterials &&
	  pMaterial.get() <= nMaterials))
      {
	reset = true;
      }

    MPMaterial *mat;
    ParticleSetHandle psh;

    if( !reset && sMaterial.get() > 0 ){
      cerr<<"before getMaterial(sMaterial.get() -1)\n";
      cerr<<" sMaterial.get() = "<< sMaterial.get() << endl;
      mat =  tpr->getMaterial(sMaterial.get() - 1);
      cerr<<"after getMaterial(sMaterial.get() -1)\n";
      if( (mat->getScalarField( sVar.get() )).get_rep() == 0 )
	reset = true;
    }

    if( !reset && vMaterial.get() > 0) {
      cerr<<"before getMaterial(vMaterial.get() -1)\n";
      mat = tpr->getMaterial(vMaterial.get() - 1);
      cerr<<"after getMaterial(vMaterial.get() -1)\n";
      if( (mat->getVectorField( vVar.get() )).get_rep() == 0 )
	reset = true;
    }

    if( !reset && pMaterial.get() > 0 ) {
      cerr<<"before getMaterial(pMaterial.get() -1)\n";
      mat = tpr->getMaterial(pMaterial.get());
      cerr<<"after getMaterial(pMaterial.get() -1)\n";
      psh = mat->getParticleSet();
      if( psh.get_rep() != 0  && ( psh->find_scalar( psVar.get() ) == -1 ||
	  psh->find_vector( pvVar.get() ) == -1 ))
	reset = true;
    }

    if( reset && nMaterials ){
      sMaterial.set(1);
      vMaterial.set(1);
      Array1<clString> vars;
      mat = tpr->getMaterial(0);
      psh = mat->getParticleSet();
      mat->getScalarNames( vars );
      sVar.set( vars[0] );
      psVar.set( vars[0] );
      vars.remove_all();
      mat->getVectorNames( vars );
      vVar.set( vars[0] );
      pvVar.set( vars[0] );

      if( psh.get_rep() != 0) 
	pMaterial.set( 1 );
    }

    clString result;
    if( nMaterials ){
      TCL::eval( id+" Visible", result);
      if( result == "1" )
	TCL::execute( id + " Rebuild" );
    }
    
  }
  NOT_FINISHED("ParticleGridVisControl::setVars()");
}

//////////
// Check to see that this data has the same variables as the last
// set of data.  If they are different then call setVars, because
// previous selections do not apply.
void ParticleGridVisControl::checkVars(ParticleGridReaderHandle reader)
{
  if(TecplotReader *tpr = dynamic_cast<TecplotReader*> (reader.get_rep()))
    if (TecplotReader *pgr = dynamic_cast<TecplotReader*>  (pgrh.get_rep())){
    int i;
    int nMaterials = tpr->getNMaterials();
    if( nMaterials != pgr->getNMaterials() ) {
      setVars(tpr);
      return;
    }
  
    for(i = 0; i < nMaterials; i++ ){
      MPMaterial *mat1 = tpr->getMaterial(i);
      MPMaterial *mat2 = pgr->getMaterial(i);
      ParticleSetHandle psh1 = mat1->getParticleSet();
      ParticleSetHandle psh2 = mat2->getParticleSet();    
      if( (psh1.get_rep() != 0 && psh2.get_rep() == 0) 
	  ||(psh1.get_rep() == 0 && psh2.get_rep() != 0)) {
	setVars(tpr);
	return;
      }
    }
    Array1< clString> str1;
    Array1< clString> str2;
    pgr->getMaterial(0)->getScalarNames( str1 );
    tpr->getMaterial(0)->getScalarNames( str2 );
    if( str1.size() != str2.size() ) {
      setVars(tpr);
      return;
    } else {
      for( i = 0; i < str1.size(); i++)
	{
	  if( str1[i] != str2[i] ) {
	    setVars(tpr);
	    return;
	  }
	}
    }
    str1.remove_all(); str2.remove_all();
    pgr->getMaterial(0)->getVectorNames( str1 );
    tpr->getMaterial(0)->getVectorNames(str2);
    if( str1.size() != str2.size() ) {
      setVars(tpr);
      return;
    } else {
      for( i = 0; i < str1.size(); i++)
	{
	  if( str1[i] != str2[i] ) {
	    setVars(tpr);
	    return;
	  }
	}
    }
  }
}

void ParticleGridVisControl::callback( int index)
{
  cerr<< "ParticleGridVisControl::callback request data for index "<<
    index << ".\n";

  clString idx = to_string(index);
  TCL::execute( id + " infoFrame " + idx);
  TCL::execute( id + " infoAdd " + idx + " " + "0" + 
		" Info for particle " + idx + " in material " +
		to_string( pMaterial.get() )+ ".\n");

  int i;
  Array1< clString> vars;
  if( TecplotReader* tpr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
    ParticleSetHandle psh =
      tpr->getMaterial( pMaterial.get() -1)->getParticleSet();
    ParticleSet *ps = psh.get_rep();
  
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

void ParticleGridVisControl::graph(clString idx, clString var)
{
  int i;
  if( TecplotReader *tpr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){

    cerr<< "ntimesteps = "<< tpr->GetNTimesteps()<<endl;
    Array1<float> values;
    if( tpr->GetNTimesteps() ){
      int varId = tpr->getMaterial( pMaterial.get() -1)->
	getParticleSet()->find_scalar( var ); // psVar.get() );
      tpr->GetParticleData(atoi(idx()), varId, pMaterial.get(), false, values);
    
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
void ParticleGridVisControl::execute() 
{ 
   tcl_status.set("Calling ParticleGridVisControl!"); 

   ParticleGridReaderHandle handle;
   if(!in->get(handle)){
     return;
   }

   MPMaterial *mat;

   if( TecplotReader *tpr = dynamic_cast<TecplotReader*> (handle.get_rep())){
   
     if ( handle.get_rep() != pgrh.get_rep() ) {
       
       if (pgrh.get_rep()  == 0 ){
	 setVars( handle );
       } else {
	 checkVars( handle );
       }
       pgrh = handle;
     }       
     
     if ( sVar.get() != ""  && sMaterial.get() > 0 ) {
       mat = tpr->getMaterial( sMaterial.get() - 1 );
       ScalarFieldHandle sfh = mat->getScalarField( clString(sVar.get()) );
       if ( sfh.get_rep() )
	 sfout->send(sfh);
     }
     if ( vVar.get() != ""  && vMaterial.get() >  0 ) {
       mat = tpr->getMaterial( vMaterial.get() - 1);
       VectorFieldHandle vfh = mat->getVectorField( clString(vVar.get()) );
       if( vfh.get_rep() )
	 vfout->send(vfh);
     }
     
     if ( pMaterial.get() > 0 ){
       mat = tpr->getMaterial( pMaterial.get() - 1);
       ParticleSetHandle psh = mat->getParticleSet();
       ParticleSet *ps = psh.get_rep();
       if( MPVizParticleSet *mpvps = dynamic_cast <MPVizParticleSet *> (ps)){
	 mpvps->SetScalarId( psVar.get());
	 mpvps->SetVectorId( pvVar.get());
	 mpvps->SetCallback( (void *) this);
	 cerr<<psVar.get()<<", "<<pvVar.get()<<", "<<endl;
       }
       if( psh.get_rep() ){
	 psout->send(psh);
       }
     }
   }
   cout << "Done!"<<endl; 
   NOT_FINISHED("ParticleGridVisControl::execute()");
} 
//--------------------------------------------------------------- 
} // end namespace CFD
} // end namespace SCI
  
