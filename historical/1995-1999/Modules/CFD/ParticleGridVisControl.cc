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

#include <Modules/CFD/ParticleGridVisControl.h>
#include <Modules/CFD/TecplotReader.h>
#include <Datatypes/MEFluid.h>
#include <Datatypes/ParticleSetExtension.h>
#include <Classlib/NotFinished.h>
#include <Malloc/Allocator.h>
#include <TCL/Histogram.h>
#include <iostream.h> 
#include <strstream.h>
#include <stdio.h>
#include <string.h>

namespace SCI {
namespace CFD {

extern "C" { 
  
  Module* make_ParticleGridVisControl(const clString& id) 
  { 
    return new ParticleGridVisControl(id); 
  } 
  
}//extern 
  

//--------------------------------------------------------------- 
ParticleGridVisControl::ParticleGridVisControl(const clString& id) 
  : Module("ParticleGridVisControl", id, Filter),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    vVar("vVar", id, this), psVar("psVar", id, this),
    pvVar("pvVar", id, this), sFluid("sFluid", id, this),
    vFluid("vFluid", id, this), pFluid("pFluid", id, this)
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=new ParticleGridReaderIPort(this, "ParticleGridReader",
				  ParticleGridReaderIPort::Atomic);
  sfout=new ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
  vfout=new VectorFieldOPort(this, "VectorField", VectorFieldIPort::Atomic);
  psout=new ParticleSetOPort(this, "ParticleSet", ParticleSetIPort::Atomic);
  pseout = new ParticleSetExtensionOPort(this, "ParticleSetExtension",
					 ParticleSetExtensionIPort::Atomic);
  // Add them to the Module
  add_iport(in);
  add_oport(sfout);
  add_oport(vfout);
  add_oport(psout);
  add_oport(pseout);

  sVar.set( "" );
  vVar.set( "" );
  psVar.set("" );
  pvVar.set( "" );

  sFluid.set(0);
  vFluid.set(0);
  pFluid.set(0);

} 

//---------------------------------------------------------- 
ParticleGridVisControl::ParticleGridVisControl(const ParticleGridVisControl& copy, int deep) 
  : Module(copy, deep),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    vVar("vVar", id, this), psVar("psVar", id, this),
    pvVar("pvVar", id, this),  sFluid("sFluid", id, this),
    vFluid("vFluid", id, this), pFluid("pFluid", id, this)

{} 

//------------------------------------------------------------ 
ParticleGridVisControl::~ParticleGridVisControl(){} 

//------------------------------------------------------------- 
Module* ParticleGridVisControl::clone(int deep) 
{ 
  return new ParticleGridVisControl(*this, deep); 
} 

//-------------------------------------------------------------- 
void ParticleGridVisControl::tcl_command( TCLArgs& args, void* userdata)
{

  int i;
  clString result;
  MEFluid *fl;
  
  if (args[1] == "getScalarVars") {
    if (args.count() != 2) {
      args.error("ParticleGridVisControl--getScalarVars");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    if ( pgrh->getNFluids() ){
      fl = pgrh->getFluid( 0 );
      Array1<clString> varnames;
      fl->getScalarVars( varnames );
      result = varnames[0];
      for( i = 1; i < varnames.size(); i++) {
	result += clString( " " + varnames[i]);
      }
      args.result( result );
    }	
  } else if (args[1] == "getVectorVars") {
    if (args.count() != 2) {
      args.error("ParticleGridVisControl--getVectorVars");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    if ( pgrh->getNFluids() ){
      fl = pgrh->getFluid( 0 );
      Array1<clString> varnames;
      fl->getVectorVars( varnames );
      result = varnames[0];
      for( i = 1; i < varnames.size(); i++) {
	result += clString( " " + varnames[i]);
      }
      args.result( result );
    }	
  } else if (args[1] == "getNFluids" ) {
    if (args.count() != 2) {
      args.error("ParticleGridVisControl--getNFluids");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    int n = pgrh->getNFluids();
    result = to_string( n );
    args.result( result );

  } else if (args[1] == "hasParticles" ) {
    if (args.count() != 3) {
      args.error("ParticleGridVisControl--hasParticles needs an index.");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    int f = atoi ( args[2]() );
    fl = pgrh->getFluid( f - 1 );
    ParticleSetHandle psh = fl->getParticleSet();
    if( psh.get_rep() == 0 )
      args.result( clString( "0" ));
    else
      args.result( clString( "1" ));
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
// the first variable in each catagory.
//
void ParticleGridVisControl::setVars(ParticleGridReaderHandle reader)
{
  int nFluids = reader->getNFluids();
  for( int i = 0; i < nFluids; i++ ){
      MEFluid *fl = reader->getFluid(i);
      ParticleSetHandle psh = fl->getParticleSet();

      if( i == 0 ) {
	sFluid.set( i + 1 );
	vFluid.set( i + 1 );
	Array1< clString > vars;
	fl->getScalarVars( vars );
	if( vars.size() > 0 ){
	  sVar.set( vars[0] );
	  psVar.set( vars[0] );
	}
	vars.remove_all();
	fl->getVectorVars( vars );
	if( vars.size() > 0 ){
	  vVar.set( vars[0] );
	  pvVar.set( vars[0] );
	}
      }
      if( psh.get_rep() != 0) {
	pFluid.set( i + 1 );
	break;
      }
  }

  clString result;
  if( nFluids ){
     TCL::eval( id+" Visible", result);
     if( result == "1" )
       TCL::execute( id + " Rebuild" );
  }
    
  NOT_FINISHED("ParticleGridVisControl::setVars()");
}

//////////
// Check to see that this data has the same variables as the last
// set of data.  If they are different then call setVars, because
// previous selections do not apply.
void ParticleGridVisControl::checkVars(ParticleGridReaderHandle reader)
{
  int i;
  ParticleGridReader *pgr = pgrh.get_rep();
  int nFluids = reader->getNFluids();
  if( nFluids != pgr->getNFluids() ) {
    setVars(reader);
    return;
  }
  
  for(i = 0; i < nFluids; i++ ){
    MEFluid *fl1 = reader->getFluid(i);
    MEFluid *fl2 = pgr->getFluid(i);
    ParticleSetHandle psh1 = fl1->getParticleSet();
    ParticleSetHandle psh2 = fl2->getParticleSet();    
    if( (psh1.get_rep() != 0 && psh2.get_rep() == 0) 
      ||(psh1.get_rep() == 0 && psh2.get_rep() != 0)) {
      setVars(reader);
      return;
    }
  }
  Array1< clString> str1;
  Array1< clString> str2;
  pgr->getFluid(0)->getScalarVars( str1 );
  reader->getFluid(0)->getScalarVars( str2 );
  if( str1.size() != str2.size() ) {
    setVars(reader);
    return;
  } else {
    for( i = 0; i < str1.size(); i++)
      {
	if( str1[i] != str2[i] ) {
	  setVars(reader);
	  return;
	}
      }
  }
  str1.remove_all(); str2.remove_all();
  pgr->getFluid(0)->getVectorVars( str1 );
  reader->getFluid(0)->getVectorVars(str2);
  if( str1.size() != str2.size() ) {
    setVars(reader);
    return;
  } else {
    for( i = 0; i < str1.size(); i++)
      {
	if( str1[i] != str2[i] ) {
	  setVars(reader);
	  return;
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
		" Info for particle " + idx + " in fluid " +
		to_string( pFluid.get() )+ ".\n");

  int i;
  Array1< clString> vars;
  ParticleSetHandle psh =  pgrh->getFluid( pFluid.get() -1)->getParticleSet();
  ParticleSet *ps = psh.get_rep();
  
  ps->list_scalars( vars );
  int timestep= 0;
  for(i = 0; i < vars.size(); i++){
    int sid = ps->find_scalar( vars[i] );
    double scale = ps->getScalar(timestep, sid, index);
    TCL::execute( id + " infoAdd " + idx + " " + to_string(pgrh->GetNTimesteps())
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

void ParticleGridVisControl::graph(clString idx, clString var)
{
  int i;
  cerr<< "ntimesteps = "<< pgrh->GetNTimesteps()<<endl;
  Array1<float> values;
  if( pgrh->GetNTimesteps() ){
    int varId = pgrh->getFluid( pFluid.get() -1)->
      getParticleSet()->find_scalar( var ); // psVar.get() );
    pgrh->GetParticleData(atoi(idx()), varId, pFluid.get(), false, values);
    
    Array1<double> vs;
    for(i = 0; i < values.size(); i++)
      vs.add( values[i] );
    
    ostrstream ostr;
    ostr << id << " graph " << idx+var<<" "<<var << " ";
    int j = 0;
    for( i = pgrh->GetStartTime(); i <= pgrh->GetEndTime();
	 i += pgrh->GetIncrement())
      {
	ostr << i << " " << values[j++] << " ";
      }
    TCL::execute( ostr.str() );
    
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

   MEFluid *fl;


   if ( handle.get_rep() != pgrh.get_rep() ) {

     if (pgrh.get_rep()  == 0 ){
       setVars( handle );
     } else {
       checkVars( handle );
     }
     pgrh = handle;

     
   }       
   
   if ( sVar.get() != ""  && sFluid.get() != 0 ) {
     fl = pgrh->getFluid( sFluid.get() - 1 );
     ScalarFieldHandle sfh = fl->getScalarField( clString(sVar.get()) );
     if ( sfh.get_rep() )
       sfout->send(sfh);
   }
   if ( vVar.get() != ""  && vFluid.get() != 0 ) {
     fl = pgrh->getFluid( vFluid.get() - 1);
     VectorFieldHandle vfh = fl->getVectorField( clString(vVar.get()) );
     if( vfh.get_rep() )
       vfout->send(vfh);
   }

   if ( pFluid.get() != 0 ){
     fl = pgrh->getFluid( pFluid.get() - 1);
     ParticleSetHandle psh = fl->getParticleSet();
     cerr<<psVar.get()<<", "<<pvVar.get()<<", "<<endl;
     ParticleSetExtension *pse =
       new ParticleSetExtension(psVar.get(), pvVar.get(), (void *) this);
     ParticleSetExtensionHandle pseh( pse );
     if( psh.get_rep() ){
       psout->send(psh);
       pseout->send(pseh);
     }
   }

   cout << "Done!"<<endl; 
   NOT_FINISHED("ParticleGridVisControl::execute()");
} 
//--------------------------------------------------------------- 
} // end namespace CFD
} // end namespace SCI
  
