/****************************************
CLASS
    VisControl

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
#include "VisControl.h"

#include <Kurt/DataArchive/VisParticleSet.h>
#include <Kurt/DataArchive/VisParticleSetPort.h>
#include <SCICore/Util/NotFinished.h>
#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/TypeDescription.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h>
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace Kurt {
namespace Modules {

using SCICore::Containers::to_string;
using Uintah::DataArchive;
using Uintah::TypeDescription;
using Uintah::Level;
using Uintah::Region;
using Uintah::Grid;
using Uintah::GridP;
using namespace SCICore::TclInterface;
using Kurt::Datatypes::VisParticleSet;
using PSECore::Datatypes::VisParticleSetOPort;
using PSECore::Datatypes::VisParticleSetIPort;
using SCICore::Geometry::BBox;
using SCICore::Datatypes::VectorFieldRG;
using SCICore::Datatypes::ScalarFieldRGdouble;

extern "C" Module* make_VisControl( const clString& id ) {
  return scinew VisControl( id ); 
}

//--------------------------------------------------------------- 
VisControl::VisControl(const clString& id) 
  : Module("VisControl", id, Filter),
    tcl_status("tcl_status", id, this), gsVar("gsVar", id, this),
    gvVar("gvVar", id, this), psVar("psVar", id, this),
    pvVar("pvVar", id, this), ptVar("ptVar", id, this),
    gtVar("gtVar", id, this), time("time", id, this),
    archive(0), positionName("")
{ 
  //////////// Initialization code goes here
  // Create Ports
  in=new ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  sfout=new ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
  vfout=new VectorFieldOPort(this, "VectorField", VectorFieldIPort::Atomic);
  psout=new VisParticleSetOPort(this, "ParticleSet", VisParticleSetIPort::Atomic);

  // Add them to the Module
  add_iport(in);
  add_oport(sfout);
  add_oport(vfout);
  add_oport(psout);
  //add_oport(pseout);

} 

//------------------------------------------------------------ 
VisControl::~VisControl(){} 

//------------------------------------------------------------- 
/*
void VisControl::tcl_command( TCLArgs& args, void* userdata)
{

  int i,j;
  clString result;
  std::cerr<<"tcl_command arg: "<< args[1]<<endl;
  
  if( args[1] == "getParticleSetScalarNames"){
    if(args.count() != 3 ) {
      args.error("VisControl--getParticleSetScalarNames");
      return;
    }


  } else if( args[1] == "getParticleSetVectorNames"){
    if(args.count() != 3 ) {
      args.error("VisControl--getParticleSetVectorNames");
      return;
    }

  } else if( args[1] == "getGridScalarNames"){
    if(args.count() != 3 ) {
      args.error("VisControl--getGridScalarNames");
      return;
    }
  } else if( args[1] == "getGridVectorNames"){
    if(args.count() != 3 ) {
      args.error("VisControl--getGridVectorNames");
      return;
    }
  } else if (args[1] == "graph" ) {
    clString var = args[3];
    clString idx = args[2];
    //    graph(idx, var);
  } else {
    Module::tcl_command(args, userdata);
  }
}
*/

//////////
// If receiving data for the first time, this function selects
// the first variable in each catagory.  If a previously save script
// has been loaded, the first data to be recieved will appear new even
// though the data being viewed is probably the same as that being 
// viewed at some previous time.  Several "if" statements are used
// to determine if the previously selected variables can still be 
// used.
//
  void VisControl::setVars(ArchiveHandle ar)
{
  string command;
  DataArchive& archive = *((*(ar.get_rep()))());

  vector< string > names;
  vector< const TypeDescription *> types;
  archive.queryVariables(names, types);

  vector< double > times;
  vector< int > indices;
  archive.queryTimesteps( indices, times );

  string psNames("");
  string pvNames("");
  //  string ptNames;
  string gsNames("");
  string gvNames("");
  //  string gtNames;

  const TypeDescription *td;
  for( int i = 0; i < names.size(); i++ ){
    td = types[i];
    if( td->getType() ==  TypeDescription::NCVariable){
       const TypeDescription* subtype = td->getSubType();
      if( subtype->getType() == TypeDescription::double_type){
	 gsNames += " ";
	gsNames += names[i];
	if( gsVar.get() == "" ){ gsVar.set( names[i].c_str() ); }
      }	else if(subtype->getType() == TypeDescription::Vector) {
	 gvNames += " ";
	gvNames += names[i];
	if( gvVar.get() == "" ){ gvVar.set( names[i].c_str() ); }
      } // else {Point,Tensor,Other}
    } else if(td->getType() ==  TypeDescription::ParticleVariable){
       const TypeDescription* subtype = td->getSubType();
      if( subtype->getType() == TypeDescription::double_type){
	 psNames += " ";
	psNames += names[i];
	if( psVar.get() == "" ){ psVar.set( names[i].c_str() ); }
	cerr << "Added scalar: " << names[i] << '\n';
      }	else if(subtype->getType() == TypeDescription::Vector) {
	 pvNames += " ";
	pvNames += names[i];
	if( pvVar.get() == "" ){ pvVar.set( names[i].c_str() ); }
	cerr << "Added vector: " << names[i] << '\n';
      } else if(subtype->getType() ==  TypeDescription::Point){
	positionName = names[i];
      }// else { Tensor,Other}
    }
  }
  clString visible;
  TCL::eval(id + " isVisible", visible);
  cerr<<"isVisible = "<<visible<<endl;
  if( visible == "1"){
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");
    
    TCL::execute(id + " SetTimeRange " + to_string( times[0] )
		 + " " + to_string(times[times.size()-1])
		 + " " + to_string((int)times.size()));
    cerr << "scalars: " << psNames << '\n';
    TCL::execute(id + " setParticleScalars " + psNames.c_str());
    cerr << "vectors: " << pvNames << '\n';
    TCL::execute(id + " setParticleVectors " + pvNames.c_str());
    TCL::execute(id + " buildVarList particleSet");
    TCL::execute(id + " setGridScalars " + gsNames.c_str());
    TCL::execute(id + " setGridVectors " + gvNames.c_str());
    TCL::execute(id + " buildVarList grid");
    TCL::execute("update idletasks");
    reset_vars();
  }
}



void VisControl::callback( int index)
{
  cerr<< "VisControl::callback request data for index "<<
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

void VisControl::graph(clString idx, clString var)
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
void VisControl::execute() 
{ 
  tcl_status.set("Calling VisControl!"); 
  
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
	 TCL::execute(id + " ui");
       }
     }
     setVars( handle );
     archive = handle;
   }       
     
   cerr << "done with setVars\n";
   DataArchive& archive = *((*(handle.get_rep()))());
   double t = time.get();
   int idx = 0;
   vector< double > times;
   vector< int > indices;
   archive.queryTimesteps( indices, times );
   while(t>times[idx] && idx < times.size())
      idx++;
   if(idx >= times.size())
      idx=times.size()-1;
   cerr << "Closest time is: " << times[idx] << "\n";
   GridP grid = archive.queryGrid(times[idx]);
   LevelP level = grid->getLevel( 0 );

   // get index and spatial ranges 
   BBox indexbox;
   BBox spatialbox;
   level->getIndexRange(indexbox);
   level->getSpatialRange(spatialbox);

   VectorFieldRG* vf = new VectorFieldRG();
   ScalarFieldRGdouble* sf = new ScalarFieldRGdouble();
   // resize and set bounds
   vf->resize(indexbox.max().x() - indexbox.min().x(),
	     indexbox.max().y() - indexbox.min().y(),
	     indexbox.max().z() - indexbox.min().z());
   sf->resize(indexbox.max().x() - indexbox.min().x(),
	     indexbox.max().y() - indexbox.min().y(),
	     indexbox.max().z() - indexbox.min().z());
   vf->set_bounds(spatialbox.min(), spatialbox.max());
   sf->set_bounds(spatialbox.min(), spatialbox.max());

   
   ParticleSubset* dest_subset = new ParticleSubset();
   ParticleVariable< Vector > vectors(dest_subset);
   ParticleVariable< Point > positions(dest_subset);
   ParticleVariable< double > scalars(dest_subset);
   // iterator over regions
   for(Level::const_regionIterator r = level->regionsBegin();
       r != level->regionsEnd(); r++ ){
     NCVariable< Vector >  vv;
     NCVariable< double >  sv;
     ParticleVariable< Vector > pv;
     ParticleVariable< double > ps;
     ParticleVariable< Point  > pp;
     
     int matlIndex=0; // HARDCODED - Steve.  This should be fixed!
     if(gvVar.get() != "")
	archive.query(vv, string(gvVar.get()()), matlIndex, *r, times[idx]);
     if(gsVar.get() != "")
	archive.query(sv, string(gsVar.get()()), matlIndex, *r, times[idx]);
     if(pvVar.get() != "")
	archive.query(pv, string(pvVar.get()()), matlIndex, *r, times[idx]);
     if(psVar.get() != "")
	archive.query(ps, string(psVar.get()()), matlIndex, *r, times[idx]);
     if(positionName != "")
	archive.query(pp, positionName, matlIndex, *r, times[idx]);
     
#if 0
     // fill up the scalar and vector fields
     for(NodeIterator n = (*r)->getNodeIterator(); !n.done(); n++){
       sf->grid((*n).x(), (*n).y(), (*n).z() ) = sv[*n];
       vf->grid((*n).x(), (*n).y(), (*n).z() ) = vv[*n]; 
     }
#endif
     ParticleSubset* source_subset = ps.getParticleSubset();
     particleIndex dest = dest_subset->addParticles(source_subset->numParticles());
     vectors.resync();
     positions.resync();
     scalars.resync();
     for(ParticleSubset::iterator iter = source_subset->begin();
	 iter != source_subset->end(); iter++, dest++){
       vectors[dest]=pv[*iter];
       positions[dest]=pp[*iter];
       scalars[dest]=ps[*iter];
     }
   }
   VisParticleSet*vps = new VisParticleSet(positions, scalars, vectors, this);
   
   sfout->send( sf );
   vfout->send( vf );
   psout->send( vps );
   cerr << "all done\n";
} 
//--------------------------------------------------------------- 
} // end namespace Modules
} // end namespace Kurt
  
