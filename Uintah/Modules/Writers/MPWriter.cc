 //static char *id="@(#) $Id$";

/****************************************
CLASS
    MPWriter

    Visualization control for simulation data that contains
    information on both a regular grid in particle sets.

OVERVIEW TEXT
    This module receives a MPWriterReader object.  The user
    interface is dynamically created based information provided by the
    MPWriterReader.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    MPWriterReader, Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Malloc/Allocator.h>

#include <Uintah/Datatypes/Particles/cfdlibParticleSet.h>
#include <Uintah/Datatypes/Particles/MPVizParticleSet.h>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
using std::ios;
#include <fstream>
#include <sstream>

#include <Uintah/Datatypes/Particles/MPWrite.h>
#include "MPWriter.h"
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <Uintah/Modules/MPMViz/TecplotReader.h>

namespace Uintah {
namespace Modules {

using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
 
PSECore::Dataflow::Module*
  make_MPWriter( const clString& id )
  { 
    return new MPWriter( id ); 
  } 

//--------------------------------------------------------------- 
MPWriter::MPWriter(const clString& id) 
  : Module("MPWriter", id, Filter),
    tcl_status("tcl_status", id, this), sVar("sVar", id, this),
    vVar("vVar", id, this), psVar("psVar", id, this),
    pvVar("pvVar", id, this), sMaterial("sMaterial", id, this),
    vMaterial("vMaterial", id, this), pMaterial("pMaterial", id, this)
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=new ParticleGridReaderIPort(this, "ParticleGridReader",
				  ParticleGridReaderIPort::Atomic);
  //  pseout = new ParticleSetExtensionOPort(this, "ParticleSetExtension",
  //				 ParticleSetExtensionIPort::Atomic);
  // Add them to the Module
  add_iport(in);
  //  add_oport(pseout);

} 


//------------------------------------------------------------ 
MPWriter::~MPWriter(){} 

//------------------------------------------------------------- 
void MPWriter::tcl_command( TCLArgs& args, void* userdata)
{

  int i;
  clString result;
  MPMaterial *mat;
  
  if (args[1] == "getScalarNames") {
    if (args.count() != 2) {
      args.error("MPWriter--getScalarNames");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    if( TecplotReader* tr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      if ( tr->getNMaterials() ){
	mat = tr->getMaterial( 0 );
	Array1<clString> varnames;
	mat->getScalarNames( varnames );
	result = varnames[0];
	for( i = 1; i < varnames.size(); i++) {
	  result += clString( " " + varnames[i]);
	}
	args.result( result );
      } else
	return;
    }
      
  } else if (args[1] == "getVectorNames") {
    if (args.count() != 2) {
      args.error("MPWriter--getVectorNames");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    if( TecplotReader* tr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      if ( tr->getNMaterials() ){
	mat = tr->getMaterial( 0 );
	Array1<clString> varnames;
	mat->getVectorNames( varnames );
	result = varnames[0];
	for( i = 1; i < varnames.size(); i++) {
	  result += clString( " " + varnames[i]);
	}
	args.result( result );
      }	else 
	return;
    }

  } else if (args[1] == "getNMaterials" ) {
    if (args.count() != 2) {
      args.error("MPWriter--getNMaterials");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    if( TecplotReader* tr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      int n = tr->getNMaterials();
      result = to_string( n );
      args.result( result );
    }
  } else if (args[1] == "hasParticles" ) {
    if (args.count() != 3) {
      args.error("MPWriter--hasParticles needs an index.");
      return;
    }
    if( pgrh.get_rep() == 0 ) {
      return;
    }
    if( TecplotReader* tr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
      int f = atoi ( args[2]() );
      mat = tr->getMaterial( f - 1 );
      ParticleSetHandle psh = mat->getParticleSet();
      if( psh.get_rep() == 0 )
	args.result( clString( "0" ));
      else
	args.result( clString( "1" ));
    } else {
      return;
    }
  } else if (args[1] == "save") {
    if (args.count() != 4){
      args.error("MPWriter--save needs a file name and binary flag");
      return;
    }
    int isBin = atoi( args[3]() );
    SaveFile( args[2], isBin );
  } else {
    Module::tcl_command(args, userdata);
  }
}


void MPWriter::SaveFile( clString fname, int isBin)
{
  int i,j,k;
  clString isBinFile;
  if( isBin ) isBinFile = "BIN";
  else isBinFile = "ASCII";
  MPWrite writer( fname, ios::out );

  if( TecplotReader* tr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
    MPMaterial *mat = tr->getMaterial( 0 );
    Array1<clString> svarnames;
    mat->getScalarNames( svarnames );
    clString sresult = svarnames[0];
    for( i = 1; i < svarnames.size(); i++) {
      sresult += clString( " " + svarnames[i]);
    }
    Array1<clString> vvarnames;
    mat->getVectorNames( vvarnames );
    clString vresult = vvarnames[0];
    for( i = 1; i < vvarnames.size(); i++) {
      vresult += clString( " " + vvarnames[i]);
    }
    clString mresult("");
    for (i = 0; i < tr->getNMaterials(); i++)
      mresult += clString(" " + to_string(i+1));
  
  
    writer.BeginHeader( "Test",
			isBinFile,
			" Testing comments ");
    writer.AddComment(" another test ");
    writer.EndHeader();
  
    // Now get size of scalar field
    VectorFieldHandle vfh = tr->getMaterial(0)->getVectorField(vvarnames[0]);
    VectorField *vfp = vfh.get_rep();
    if( VectorFieldRG * vfrg = dynamic_cast<VectorFieldRG*> (vfp) ){
      Array1< double > X;
      Array1< double > Y;
      Array1< double > Z;
      
      for( i = 0; i < vfrg->nx; i++)
	X.add( vfrg->get_point(i,0,0).x());
      for(j = 0; j < vfrg->ny; j++)
	Y.add(vfrg->get_point(0,j,0).y());
      for( k = 0; k < vfrg->nz; k++)
	Z.add(vfrg->get_point(0,0,k).z());
    
      writer.BeginGrid( "collision", "NC",
			sresult, vresult,
			X.size(), Y.size(), Z.size(),
			X[0], Y[0], Z[0],
			(X[X.size()-1]-X[0])/(X.size() - 1),
			(Y[Y.size()-1]-Y[0])/(Y.size() - 1),
			(Z[Z.size()-1]-Z[0])/(Z.size() - 1));
    }
    for( j = 0; j < tr->getNMaterials(); j++){
      mat = tr->getMaterial(j);
      svarnames.remove_all();
      mat->getScalarNames( svarnames );
      for( i = 0; i < svarnames.size(); i++){
	ScalarFieldHandle sfh = mat->getScalarField( svarnames[i] );
	ScalarField *sfp = sfh.get_rep();
	writer.AddSVarToGrid(svarnames[i], sfp);
      }
    }
  
    for( j = 0; j < tr->getNMaterials(); j++){
      mat = tr->getMaterial(j);
      vvarnames.remove_all();
      mat->getVectorNames( vvarnames );
      for( i = 0; i < vvarnames.size(); i++){
	vfh = mat->getVectorField( vvarnames[i] );
	VectorField *vfp = vfh.get_rep();
	if(VectorFieldRG *vfrg = dynamic_cast<VectorFieldRG*> (vfp)){
	  writer.AddVecVarToGrid(vvarnames[i], vfrg);
	} else {
	  cerr<<"Error: unknown grid type\n";
	  return;
	}
      }
    }

    writer.EndGrid();

    Array1<clString> snames;
    Array1<clString> vnames;
    clString sn("");
    clString vn("");
    Array1<Vector> pos;
    int posid = 0, sid, vid;
    for( k = 0; k < tr->getNMaterials(); k++){
      mat = tr->getMaterial(k);
      ParticleSetHandle psh = mat->getParticleSet();
      ParticleSet *ps = psh.get_rep();
      if(cfdlibParticleSet* cfdps = dynamic_cast< cfdlibParticleSet*> (ps)){

	Array1< double> s;
	Array1< Vector> v;

	cfdps->get(0,0,s);
	posid = cfdps->position_vector();
	cfdps->get(0, posid, pos);
	vid = 1; sid = 0;      cfdps->list_scalars(snames);
	cfdps->list_vectors(vnames);
	for(i = sid; i < snames.size(); i++){
	  sn += (snames[i]);
	  sn += " ";
	}
	for(i = vid; i < vnames.size(); i++){
	  vn += (vnames[i]);
	  vn += " ";
	}
	writer.BeginParticles("steel", s.size(),  sn, vn );

	for(j = 0; j < pos.size(); j++){
	  s.remove_all();
	  v.remove_all();

	  for(i = sid; i < snames.size(); i++) {
	    s.add( cfdps->getScalar(0,i,j));
	  }
	  for(i = vid; i < vnames.size(); i++) {
	    v.add(cfdps->getVector(0,i,j));
	  }
	  writer.AddParticle( pos[j].asPoint(), s, v);
	}
      }
    }
    writer.EndParticles();
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
void MPWriter::setVars(ParticleGridReaderHandle reader)
{
  if( TecplotReader* r = dynamic_cast<TecplotReader*>( reader.get_rep())){
    int nMaterials = r->getNMaterials();
    bool reset = false;
    if (!(sMaterial.get() <= nMaterials &&
	  vMaterial.get() <= nMaterials &&
	  pMaterial.get() <= nMaterials))
      {
	reset = true;
      }

    MPMaterial *mat;
    ParticleSetHandle psh;

    if( !reset ){
      mat =  r->getMaterial(sMaterial.get() - 1);
      if( (mat->getScalarField( sVar.get() )).get_rep() == 0 )
	reset = true;
    }

    if( !reset ) {
      mat = r->getMaterial(vMaterial.get() - 1);
      if( (mat->getVectorField( vVar.get() )).get_rep() == 0 )
	reset = true;
    }

    if( !reset ) {
      mat = r->getMaterial(pMaterial.get() - 1);
      psh = mat->getParticleSet();
      if( psh->find_scalar( psVar.get() ) == -1 ||
	  psh->find_vector( pvVar.get() ) == -1 )
	reset = true;
    }

    if( reset && nMaterials ){
      sMaterial.set(1);
      vMaterial.set(1);
      Array1<clString> vars;
      mat = r->getMaterial(0);
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
  NOT_FINISHED("MPWriter::setVars()");
}

//////////
// Check to see that this data has the same variables as the last
// set of data.  If they are different then call setVars, because
// previous selections do not apply.
void MPWriter::checkVars(ParticleGridReaderHandle reader)
{
  int i;
  if( TecplotReader* tr = dynamic_cast<TecplotReader*> (pgrh.get_rep())){
    if( TecplotReader* r = dynamic_cast<TecplotReader*> (reader.get_rep())){
      int nMaterials = r->getNMaterials();
      if( nMaterials != tr->getNMaterials() ) {
	setVars(r);
	return;
      }
  
      for(i = 0; i < nMaterials; i++ ){
	MPMaterial *mat1 = r->getMaterial(i);
	MPMaterial *mat2 = tr->getMaterial(i);
	ParticleSetHandle psh1 = mat1->getParticleSet();
	ParticleSetHandle psh2 = mat2->getParticleSet();    
	if( (psh1.get_rep() != 0 && psh2.get_rep() == 0) 
	    ||(psh1.get_rep() == 0 && psh2.get_rep() != 0)) {
	  setVars(r);
	  return;
	}
      }
      Array1< clString> str1;
      Array1< clString> str2;
      tr->getMaterial(0)->getScalarNames( str1 );
      r->getMaterial(0)->getScalarNames( str2 );
      if( str1.size() != str2.size() ) {
	setVars(r);
	return;
      } else {
	for( i = 0; i < str1.size(); i++)
	  {
	    if( str1[i] != str2[i] ) {
	      setVars(r);
	      return;
	    }
	  }
      }
      str1.remove_all(); str2.remove_all();
      tr->getMaterial(0)->getVectorNames( str1 );
      r->getMaterial(0)->getVectorNames(str2);
      if( str1.size() != str2.size() ) {
	setVars(r);
	return;
      } else {
	for( i = 0; i < str1.size(); i++)
	  {
	    if( str1[i] != str2[i] ) {
	      setVars(r);
	      return;
	    }
	  }
      }
    }
  }
}


//----------------------------------------------------------------
void MPWriter::execute() 
{ 
   tcl_status.set("Calling MPWriter!"); 

   ParticleGridReaderHandle handle;
   if(!in->get(handle)){
     return;
   }


   if ( handle.get_rep() != pgrh.get_rep() ) {
     pgrh = handle;
   }       
   

   cout << "Done!"<<endl; 
   NOT_FINISHED("MPWriter::execute()");
} 
//--------------------------------------------------------------- 
} // end namespace CFD
} // end namespace SCI
  
