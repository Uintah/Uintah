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
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 5, 1999
****************************************/
#include "VisControl.h"

#include <Packages/Kurt/DataArchive/VisParticleSet.h>
#include <Packages/Kurt/DataArchive/VisParticleSetPort.h>
#include <Core/Util/NotFinished.h>
#include <Packages/Uintah/Interface/DataArchive.h>
#include <Packages/Uintah/Grid/TypeDescription.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Packages/Uintah/Interface/DataArchive.h>
#include <Packages/Uintah/Grid/Grid.h>
#include <Packages/Uintah/Grid/GridP.h>
#include <Packages/Uintah/Grid/Level.h>
#include <Packages/Uintah/Grid/Patch.h>
#include <Packages/Uintah/Grid/NodeIterator.h>
#include <iostream> 
#include <sstream>
#include <string>

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace Kurt {

using namespace SCIRun;
using namespace Uintah;

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
    timeval("timeval", id, this),
    gsMatNum("gsMatNum", id, this),
    gvMatNum("gvMatNum", id, this),
    pNMaterials("pNMaterials", id, this),
    animate("animate",id, this),archive(0), positionName("")
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
  
  // reset the vars
  psVar.set("");
  pvVar.set("");
  gsVar.set("");
  gvVar.set("");

  // get all of the NC and Particle Variables
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

  // get the number of materials for the NC & particle Variables
  GridP grid = archive.queryGrid(times[0]);
  LevelP level = grid->getLevel( 0 );
  Patch* r = *(level->patchesBegin());
  int numpsMatls = archive.queryNumMaterials(psVar.get()(), r, times[0]);
  int numgsMatls = archive.queryNumMaterials(gsVar.get()(), r, times[0]);
  int numgvMatls = archive.queryNumMaterials(gvVar.get()(), r, times[0]);

  clString visible;
  TCL::eval(id + " isVisible", visible);
  if( visible == "1"){
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");
    
    TCL::execute(id + " buildPMaterials " + to_string(numpsMatls));
    pNMaterials.set(numpsMatls);
    TCL::execute(id + " buildGsGvMaterials " 
		 + to_string(numgsMatls) + " " + to_string(numgvMatls));

    TCL::execute(id + " SetTimeRange " + to_string((int)times.size()));
    TCL::execute(id + " setParticleScalars " + psNames.c_str());
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
	 TCL::execute(id + " buildTopLevel");
       }
     }
     setVars( handle );
     archive = handle;
   }       
     
   cerr << "done with setVars\n";



   DataArchive& archive = *((*(handle.get_rep()))());
   VectorFieldRG* vf;
   ScalarFieldRGdouble* sf;
   VisParticleSet* vps;

   // what time is it?
   
   int t = time.get();

   // set the index for the correct timestep.
   int idx = 0;
   vector< double > times;
   vector< int > indices;
   archive.queryTimesteps( indices, times );
   if(t < times.size())
     idx = t;
   if(t >= times.size())
     idx=times.size()-1;

   timeval.set(times[idx]);

   if( animate.get() ){
     while( animate.get() && idx < times.size() - 1){
       tcl_status.set( to_string( times[idx] ));
       time.set( times[idx] );
       buildData( archive, times, idx , sf, vf, vps );
       sfout->send_intermediate( sf );
       vfout->send_intermediate( vf );
       psout->send_intermediate( vps );
       reset_vars();
       idx++;
     }
     animate.set(0);
     reset_vars();
   }
   buildData( archive, times, idx , sf, vf, vps );
   sfout->send( sf );
   vfout->send( vf );
   psout->send( vps );	  
   tcl_status.set("Done");
}

void 
VisControl::buildData(DataArchive& archive, vector< double >& times,
		      int idx, ScalarFieldRGdouble*& sf,
		      VectorFieldRG*& vf, VisParticleSet*& vps)
{
  
  cerr << "Closest time is: " << times[idx] << "\n";
  GridP grid = archive.queryGrid(times[idx]);
  LevelP level = grid->getLevel( 0 );

  // get index and spatial ranges 
  BBox spatialbox;
  IntVector minIV, maxIV;
  level->getIndexRange(minIV, maxIV);
  level->getSpatialRange(spatialbox);

  vf = new VectorFieldRG();
  sf = new ScalarFieldRGdouble();

  // resize and set bounds
  vf->resize(maxIV.x() - minIV.x(),
	     maxIV.y() - minIV.y(),
	     maxIV.z() - minIV.z());
  sf->resize(maxIV.x() - minIV.x(),
	     maxIV.y() - minIV.y(),
	     maxIV.z() - minIV.z());
  vf->set_bounds(spatialbox.min(), spatialbox.max());
  sf->set_bounds(spatialbox.min(), spatialbox.max());
  std::cerr<< "Bounds = ( "<<spatialbox.min()<<", "<<spatialbox.max()<<
    " )\n";


   
  ParticleSubset* dest_subset = new ParticleSubset();
  ParticleVariable< Vector > vectors(dest_subset);
  ParticleVariable< Point > positions(dest_subset);
  ParticleVariable< double > scalars(dest_subset);

  // iterate over patches
  for(Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); r++ ){
    NCVariable< Vector >  vv;
    NCVariable< double >  sv;
    ParticleVariable< Vector > pv;
    ParticleVariable< double > ps;
    ParticleVariable< Point  > pp;
     
    int matlIndex=gsMatNum.get();
    bool have_gv;
    if(gvVar.get() != ""){
      have_gv=true;
      archive.query(vv, string(gvVar.get()()), matlIndex, *r, times[idx]);
    } else {
      have_gv=false;
    }
    matlIndex = gvMatNum.get();
    bool have_gs;
    if(gsVar.get() != ""){
      have_gs=true;
      archive.query(sv, string(gsVar.get()()), matlIndex, *r, times[idx]);
    } else {
      have_gs=false;
    }
    // fill up the scalar and vector fields
    for(NodeIterator n = (*r)->getNodeIterator(); !n.done(); n++){
      if(have_gs)
	sf->grid((*n).x(), (*n).y(), (*n).z() ) = sv[*n];
      else
	sf->grid((*n).x(), (*n).y(), (*n).z()) = 0;
      if(have_gv)
	vf->grid((*n).x(), (*n).y(), (*n).z() ) = vv[*n]; 
      else
	vf->grid((*n).x(), (*n).y(), (*n).z() ) = Vector(0,0,0);
    }

    int numMatls = archive.queryNumMaterials(positionName, *r, times[idx]);
    bool have_pv;
    if(pvVar.get() != ""){
      have_pv=true;
    } else {
      have_pv=false;
    }
    for(int matl=0;matl<numMatls;matl++){
      clString result;
      eval(id + " isOn p" + to_string(matl), result);
      if ( result == "0")
	continue;
      if (have_pv)
	archive.query(pv, string(pvVar.get()()), matl, *r, times[idx]);
      if( psVar.get() != "")
	archive.query(ps, string(psVar.get()()), matl, *r, times[idx]);
      if(positionName != "")
	archive.query(pp, positionName, matl, *r, times[idx]);
      
      ParticleSubset* source_subset = ps.getParticleSubset();
      particleIndex dest = dest_subset->addParticles(source_subset->numParticles());
      vectors.resync();
      positions.resync();
      scalars.resync();
      for(ParticleSubset::iterator iter = source_subset->begin();
	  iter != source_subset->end(); iter++, dest++){
	if(have_pv)
	  vectors[dest]=pv[*iter];
	else
	  vectors[dest]=Vector(0,0,0);
	positions[dest]=pp[*iter];
	scalars[dest]=ps[*iter];
      }
    }
  }
  vps = new VisParticleSet(positions, scalars, vectors, this);
} 

} // End namespace Kurt
