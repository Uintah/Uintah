//static char *id="@(#) $Id$";

/****************************************
CLASS
    MPWrite

    A class for writing Material/Particle files.

OVERVIEW TEXT

KEYWORDS
    Material/Particle Method

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June 1999

    Copyright (C) 1999 SCI Group

LOG
    June 28, 1999
****************************************/



#include <fstream.h>
#include <strstream.h>
#include "MPWrite.h"

namespace Uintah {
namespace Datatypes {  

using namespace SCICore::Datatypes;
using namespace SCICore::Containers; 


MPWrite::MPWrite(clString fname, int mode): os(fname(), mode ),
  state( Closed ), currentGrid(None),
  headerWritten( false ), svCount(0), vvCount(0)
{
  if (!(os.good())) {
    cerr<<"Error in opening file: "<<fname<<endl;
  } else {
    state = Open;
  }
}

MPWrite::~MPWrite()
{
  os.close();
}

int 
MPWrite::BeginHeader(clString title,
		     clString fileType, // BIN or ASCII
		     clString comment)
{
  if(!(state == Open)){
    cerr<<"Error:  file is not open\n";
    return 0;
  } else {
    state = Header;
    os << "MPD 1.0 "<<fileType<<endl;
    os << "TITLE:  "<<title<<endl;
    os << "## "<<comment<<endl;

    if(fileType == "BIN" )
      this->fileType = BIN;
    else
      this->fileType = ASCII;
    return 1;
  }
}

// AddComment can only be called between the BeginHeader and EndHeader calls

int 
MPWrite::AddComment( clString comment )
{
  if( state != Header) {
    cerr<<"Error: can only add comments to header\n";
    return 0;
  } else {
    os << "## "<<comment<<endl;
    return 1;
  }
}
    

int 
MPWrite::EndHeader()
{
  if( state != Header ){
    cerr<<"Error: illegal action. Not writing header\n";
    return 0;
  } else {
    os << "END_HEADER\n";
    state = Open;
    headerWritten = true;
    return 1;
  }
}

void
MPWrite::printCurrentState(ostream& out)
{
  clString cs("");
  switch (state) {
  case Open:
    cs += "Open";
    break;
  case Closed:
    cs += "Closed";
    break;
  case Header:
    cs += "Header";
    break;
  case Grid:
    cs += "Grid";
    break;
  case Particles:
    cs += "Particles";
    break;
  default:
    cs += "None";
  }
  out<< "Current state is "<<cs<<".";
}

int 
MPWrite::BeginGrid( clString name,
		    clString type,
		    clString scalarVars,
		    clString vectorVars,
		    Array1< double > X,  // an array of X points
		    Array1< double > Y,
		    Array1< double > Z )
{
  if( !headerWritten ){
    cerr<<"Error:  You must first write out a header\n";
    return 0;
  } else if( state == Open ) {
    os << "GRID  " <<name<<" "<<type
       <<" "<<X.size()<<" "<<Y.size()<<" "<<Z.size()<<endl;
    os << scalarVars <<endl;
    os << vectorVars <<endl;

    if( fileType == BIN ){
      os.write((char *) &(X[0]), sizeof(double) * X.size());
      os.write((char *) &(Y[0]), sizeof(double) * Y.size());
      os.write((char *) &(Y[0]), sizeof(double) * Z.size());
    } else {
      for(int i = 0; i < X.size(); i++)	os << X[i] << " ";
      for(int i = 0; i < Y.size(); i++) os << Y[i] << " ";
      for(int i = 0; i < Z.size(); i++) os << Z[i] << " ";
      
    }
    
    istrstream sv( scalarVars() );
    istrstream vv( vectorVars() );
    char inbuf[100];
    while( sv >> inbuf ) sVars.add( clString(inbuf) );
    while( vv >> inbuf ) vVars.add( clString(inbuf) );


    cerr<<"In BeginGrid with type = " <<type<<endl;
    if( type == "NC_i") currentGrid = NC_i;
    else if( type == "CC_i") currentGrid = CC_i;
    else if( type == "FC_i") currentGrid = FC_i;
    else currentGrid = None;

    state = Grid;
    
    return 1;
  } else {
    return 0;
  }
}

int 
MPWrite::BeginGrid( clString name,
		    clString type,
		    clString scalarVars,
		    clString vectorVars,
		    double sizeX, double sizeY, double sizeZ,
		    double minX, double minY, double minZ,
		    double dx, double dy, double dz)
{
  if( !headerWritten ){
    cerr<<"Error:  You must first write out a header\n";
    return 0;
  } else if( state == Open ) {
    os << "GRID  " <<name<<" "<<type<<" " <<
      sizeX << " " << sizeY << " " << sizeZ <<endl;
    os << scalarVars <<endl;
    os << vectorVars <<endl;
    os << minX << " " << minY << " " << minZ << " ";
    os << dx << " " << dy << " " << dz << endl;

    istrstream sv( scalarVars() );
    istrstream vv( vectorVars() );
    char inbuf[100];
    while( sv >> inbuf ) sVars.add( clString(inbuf) );
    while( vv >> inbuf ) vVars.add( clString(inbuf) );

    cerr<<"In BeginGrid with type = " <<type<<endl;
    if( type == "NC") currentGrid = NC;
    else if( type == "CC") currentGrid = CC;
    else if( type == "FC") currentGrid = FC;
    else currentGrid = None;

    state = Grid;
    
    return 1;
  } else {
    return 0;
  }
}



int 
MPWrite::AddVarToGrid( clString name,  // variable name check
		       ScalarField* var )
{
  if( svCount >= sVars.size() ){
    cerr<<"Error: no more scalar values can be added.\n";
    return 0;
  }
  cerr<<"svCount = "<<svCount<<" and sVars["<< svCount <<
    "] = "<< sVars[svCount]<<endl;
  if( name != sVars[ svCount ] ) {
    cerr<<"Error: you need to add values for "<< sVars[svCount]<<" first.\n";
    return 0;
  } 

  cerr<<"currentGrid = "<<currentGrid<<endl;
  if( ScalarFieldRG* sfrg = dynamic_cast <ScalarFieldRG*> (var)){
    if( currentGrid == NC){
      svCount++;
      if( fileType == BIN ) {
	os.write((char *) &(sfrg->grid(0,0,0)),
		 sizeof(double) * sfrg->grid.dim1() * 
		 sfrg->grid.dim2() * sfrg->grid.dim3());
      } else {
	for(int i = 0; i < sfrg->grid.dim1(); i++ )
	  for(int j = 0; j < sfrg->grid.dim2(); j++)
	    for(int k = 0; k < sfrg->grid.dim3(); k++)
	      os << sfrg->grid(i,j,k)<< " ";
      }
      return 1;
    }
    else {
      return 0;
    }
  } else {
    cerr<<"Error:  ";
    printCurrentState(cerr);
    return 0;
  }
} 

int
MPWrite::AddVarToGrid(clString name, double *sf, int length)
{
  if( svCount >= sVars.size() ){
    cerr<<"Error: no more scalar values can be added.\n";
    return 0;
  }
  
  if( name != sVars[ svCount ] ) {
    cerr<<"Error: you need to add values for "<< sVars[svCount]<<" first.\n";
    return 0;
  }
  
  if( currentGrid == NC){
    svCount++;
    if( fileType == BIN ) {
      os.write((char*) sf, sizeof(double)*length);
    } else {
      for(int i = 0; i < length; i++){
	os << sf[i] << " ";
      }
    }
    return 1;
  } else {
    return 0;
  }
}


int 
MPWrite::AddVarToGrid( clString name,
		  VectorField*  var )
{
  if(!(svCount <= sVars.size())) {
    cerr<<"Error: you must write out all scalar values first.\n";
    return 0;
  }
    if( vvCount >= vVars.size() ){
    cerr<<"Error: no more vector values can be added.\n";
    return 0;
  }
  
  if( name != vVars[ vvCount ] ) {
    cerr<<"Error: you need to add values for "<< vVars[vvCount]<<" first.\n";
    return 0;
  } else if( VectorFieldRG* vfrg = dynamic_cast <VectorFieldRG *> (var)){
    if( currentGrid == NC){
      vvCount++;
      if( fileType == BIN ) {
	Vector *v = &(vfrg->grid(0,0,0));
	for(int i = 0; i < vfrg->grid.dim1()*vfrg->grid.dim2()*vfrg->grid.dim3(); i++){
	  os.write((char *) &(v[i]), sizeof(double)*3);
	} 
      } else {
	for(int i = 0; i < vfrg->grid.dim1(); i++ )
	  for(int j = 0; j < vfrg->grid.dim2(); j++)
	    for(int k = 0; k < vfrg->grid.dim3(); k++){
	      Vector v = vfrg->grid(i,j,k);
	      os<< v.x()<< " "<< v.y() << " " << v.z() << " ";
	    }
      }
      return 1;
    } else {
      return 0;
    }
  } else {
    cerr<<"Error: Unknown Grid type\n";
    return 0;
  }
}

int 
MPWrite::AddVarToGrid( clString name,
		       Vector* vf, int length)
{
  if(!(svCount <= sVars.size())) {
    cerr<<"Error: you must write out all scalar values first.\n";
    return 0;
  }
    if( vvCount >= vVars.size() ){
    cerr<<"Error: no more vector values can be added.\n";
    return 0;
  }
  
  if( name != vVars[ vvCount ] ) {
    cerr<<"Error: you need to add values for "<< vVars[vvCount]<<" first.\n";
    return 0;
  }
  if( currentGrid == NC){
    vvCount++;
    if( fileType == BIN ) {
      for(int i = 0; i < length; i++){
	os.write((char *) &(vf[i]), sizeof(double)*3);
      }
    } else {
      for(int i = 0; i < length; i++ )
	os<< vf[i].x()<< " "<< vf[i].y() << " " << vf[i].z() << " ";
    }
    return 1;
  } else {
    cerr<<"Error: Unknown Grid type\n";
    return 0;
  }
}


int 
MPWrite::EndGrid()
{
  os<<endl;
  state = Open;
  svCount = 0;
  vvCount = 0;
  return 1;
}


int 
MPWrite::BeginParticles( clString name, 
			 int N,  // number of particles
			 clString scalarVars,
			 clString vectorVars)
{
  cerr<<"State = "<<state<<endl;
  // This will not allow you to add more than N Particles.
  if( !headerWritten ){
    cerr<<"Error:  You must first write out a header\n";
    return 0;
  } else if( state == Open ) {
    cerr << "PARTICLES  "<<name<<" "<<N<<endl;
    os << "PARTICLES  "<<name<<" "<<N<<endl;
    os <<scalarVars<<endl;
    os <<vectorVars<<endl;

    istrstream sv( scalarVars() );
    istrstream vv( vectorVars() );
    char inbuf[100];
    while( sv >> inbuf ) psVars.add( clString(inbuf) );
    while( vv >> inbuf ) pvVars.add( clString(inbuf) );

    state = Particles;
    pCount = 0;
    pN = N;
    return 1;
  } else {
    cerr<<"Error:  ";
    printCurrentState(cerr);
    cerr<<endl;
    return 0;
  }
}
  
int 
MPWrite::AddParticle( Point p, // position
	     Array1< double >& scalars, // must correspond to scalarVars
	     Array1< Vector >& vectors) // must correspond to vectorVars
{
  int i,j;
  if( state != Particles ){
    cerr<<"Error: not writing particles. ";
    printCurrentState(cerr);
    cerr<<endl;
    return 0;
  }
  if( scalars.size() != sVars.size() ){
    cerr<<"Error: scalar array does not correspond with scalar vars.\n";
    return 0;
  }
  if( vectors.size() != vVars.size() ){
    cerr<<"Error: vector array does not correspond with vector vars.\n";
    return 0;
  }
  if (pCount < pN){
    pCount++;
    if(fileType == BIN){
      os.write((char *) &p, sizeof(double)*3);
      os.write((char *) &(scalars[0]), sizeof(double)*scalars.size());
      for(j = 0; j < vectors.size(); j++){
	os.write((char *) &(vectors[j]), sizeof(double) * 3);
      }
    } else {
      os << p.x() << " " << p.y()<<" "<<p.z()<<" ";
      for( i = 0; i < scalars.size(); i++) os << scalars[i]<< " ";
      for( i = 0; i < vectors.size(); i++) {
	os << vectors[i].x()<< " ";
	os << vectors[i].y()<< " ";
	os << vectors[i].z()<< " ";
      }
    }
    return 1;
  } else {
    cerr<<"Error:  too many points.\n";
    return 0;
  }
}

int
MPWrite::EndParticles()
{
  // if you add less than N particles this will pad the
  // the file with empty particles.
  int i,j;
  double v[3] = { 0.0,0.0,0.0 };
  double s = 0.0;
  while(pCount < pN){
    pCount++;
    if ( fileType == BIN ){
      for(i = 0; i < sVars.size(); i++)
	os.write((char *) &s, sizeof(double));
      for(j = 0; j < vVars.size(); j++){
	os.write((char*) v, sizeof(double)*3);
      }
    } else {
      for(i = 0; i < sVars.size(); i++) os << s << " ";
      for(i = 0; i < vVars.size(); i++) {
	os << v[0] << " ";
	os << v[1] << " ";
	os << v[2] << " ";
      }
    }
  }
  os << endl;
  state = Open;
  return 1;
}


} // end namespace Modules
} // end namespace Uintah
