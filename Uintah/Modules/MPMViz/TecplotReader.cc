//static char *id="@(#) $Id$";

/*
 *  TecplotReader.cc: 
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <iostream>
using std::cerr;
using std::cout;
using std::ios;
using std::endl;
#include <fstream>
using std::istream;
using std::ifstream;
#include <iomanip>
using std::setw;
#include <sstream>
using std::istringstream;
using std::ostringstream;
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>

#include <Uintah/Datatypes/Particles/MPVizParticleSet.h>
#include "TecplotReader.h"

namespace Uintah {
namespace Datatypes {

using SCICore::Containers::basename;
using SCICore::Containers::pathname;
using SCICore::Containers::to_string;

static Persistent* maker()
{
    return scinew TecplotReader();
}

PersistentTypeID TecplotReader::type_id("TecplotReader",
					"ParticleGridReader", maker);

TecplotReader::~TecplotReader()
{
  for(int i = 0; i < materials.size(); i++ )
    {
      delete materials[i];
    }
  
}

TecplotReader::TecplotReader() : xmin(0), xmax(1), ymin(1), ymax(1),
  zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0)
{
}

TecplotReader::TecplotReader(const clString& filename, int start,
			     int end, int incr) : xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime( start ),
  endTime( end ), increment(incr), filename(filename)
{
  readfile();
}

TecplotReader::TecplotReader(const clString& file ) : xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0), filename(file)
{
  readfile();
}


void TecplotReader::SetFile(const clString& file )
{
  filename = file;
}

clString TecplotReader::GetFile()
{
  return filename;
}

int TecplotReader::GetNTimesteps()
{
  return endTime - startTime;
}

MPMaterial* TecplotReader::getMaterial(int i)
{
  if( i < materials.size())
    return materials[i];
  else return 0;
}

int TecplotReader::getNMaterials()
{
  return materials.size();
}
void TecplotReader::GetParticleData(int particleId,
				    clString pSetName,
                                    clString varname,
                                    Array1<double>& values)
{
  NOT_FINISHED("TecplotReader::GetParticleData");
}

void TecplotReader::GetParticleData(int particleId,
				    int variableId,
				    int materialId,
				    bool isVector,
				    Array1<float>&  values)
{
  if(  endTime - startTime <= 0 )
    return;

  values.remove_all();
  
  float val;
  int index = 1;
  char num[2];
  int materialIndex;
  // Begin by finding the root file name;
  clString file = basename(  filename );
  clString path = pathname( filename  );
  const char *p = file();
   char n[5];
  char root[ 80 ];
  int i, ii;
  int j = 0;
  int k = 0;
  for( i= 0; i < file.len(); i++ )
    {
      if(isdigit(*p)) n[j++] = *p;
      else root[k++] = *p;
      p++;
    }
  root[k] = '\0';

  // before we begin we must determine the variable index in the tecplot 
  // file.  variableId gives us the index to the nth vector or scalar.
  // by checking isVector we know to which variableId refers.
  index = ComputeAdjustedIndex( variableId, materialId, isVector);
  cerr<<"the variable id is "<<variableId<<endl;
  cerr<<"the actual index is "<<index<<endl;
  if(index == -1) return;

  for( ii = startTime; ii <= endTime; ii += increment){
    ostringstream ostr;
    ostr.fill('0');
    ostr << path << "/"<< root<< setw(4)<<ii;
    
    ifstream in( ostr.str().c_str(), ios::in);
    if( !in ) {
      cerr << "Error opening file " << filename << endl;
      return;
    }
    char buf [LINEMAX];
    clString token;

    while( in >> token){
      if( token != "ZONE" ){
	in.getline(buf, LINEMAX);
	continue;
      } else { 
	  if( in.getline(buf, LINEMAX) ){
	    if( isBlock( buf ) ) {
	      continue;
	    } else {
	      i = find( 'I', buf);
	      if (i < particleId) return;
	      p = buf;
	      while ( *p != '"') p++;
	      while ( !isdigit(*p) && *p != '\0' && *p != '"') p++;
	      if( *p == '\0' ) cerr<<"Error in Particle format\n";
	      else if( *p == '"' ){
		//cerr<<"Error in Particle format\n";
		num[0] = '1';
	      } else {
		num[0] = *p;
	      }
	      num[1] = '\0';
	      materialIndex = atoi(num);
	      if ( materialIndex != materialId ) {
		continue;
	      } else {
		for( j = 0; j < particleId; j++ ) {
		  for(k = 0; k < variables.size(); k++) {
		    in >> val;
		  }
		}
		for(k = 0; k < variables.size(); k++){
		  in >> val;
		  if ( index == k )
		    values.add(val);
		}		
		
	      }
	    }
	  }
      }
    }
  }
  //  for( i =0; i < values.size(); i++)
  //  cerr<< values[i]<<", ";
  //cerr<<endl;
  NOT_FINISHED("TecplotReader::GetParticleData");
}

int 
TecplotReader::ComputeAdjustedIndex(int variableId, int materialId, bool isVector)
{
  int v, rs, rv, materialIndex = 1;
  int readVectors[20];
  int readScalars[20];
  clString var, stripped;
  rs = 0;
  rv = 0;

  for( v = 0; v < variables.size(); v++)
    {
      var = variables[v];
      const char *p = var();
      stripVar( var, stripped, materialIndex );
      if( *p == 'P'){
	if( variableId == 0 && !isVector)
	  return 0;
	else {
	  readScalars[rs++] = v;
	  v++;
	}
      } else if( *p == 'X'){
	readVectors[rv++] = v;
	if( TwoD )  v++; else  v += 2;
      } else if( *p == 'U' ) {
	readVectors[rv++] = v;
	if( TwoD )  v++; else  v += 2;
      } else if( *p == 'A' ) {
	readVectors[rv++] = v;
	if( TwoD )  v++; else  v += 2;
      } else if( *p == 'M' && *(++p) == 'O' ) {
	readVectors[rv++] = v;
	if( TwoD )  v++; else  v += 2;
      } else if( *p == 'S' && *(++p) == 'N') {
	readVectors[rv++] = v;
	if( TwoD )  v++; else  v += 2;
      } else {   // a scalar variable
	readScalars[rs++] = v;
      }

      if( isVector && rv > variableId && materialId == materialIndex)
	return readVectors[ variableId ];
      
      if( !isVector && rs > variableId && materialId == materialIndex)
	return readScalars[ variableId ];
    }
  return -1;
}

#define TECPLOTREADER_VERSION 1

void TecplotReader::io(Piostream& stream)
{
  stream.begin_class("TecplotReader", TECPLOTREADER_VERSION);
  ParticleGridReader::io(stream);

  stream.end_class();
}


int TecplotReader::readfile()
{
  //NOT_FINISHED("TecplotReader::readfile");

  ifstream in;
  in.open( filename(), ios::in );
  if( !in ) {
    cerr << "Error opening file " << filename << endl;
    return 0;
  }

  char buf [LINEMAX];
  clString token;


  while( in >> token ){
    if ( token == "TITLE" ) {
      in.getline(buf, LINEMAX);
      // ignore title
      cout << token <<endl;
    } else if ( token == "TEXT" ) {
      in.getline(buf, LINEMAX);
      // ignore text
      cout << token <<endl;
    } else if ( token == "T") {
      in.getline(buf, LINEMAX);
      // ignore more text
      cout << token <<endl;
    } else if ( token == "VARIABLES" ) {
      cout << token <<endl;
      readVars(in);
    } else if ( token == "ZONE" ){
      readZone(in);
    }
  }
  in.close();
  return 1;
}

void TecplotReader::readVars(istream& is)
{
  char c;
  char buf[LINEMAX];
  char tok[VARNAMELEN];
  clString token;
  int nMaterials = 1;

  while( is.getline(buf,LINEMAX) ){
    int readin = is.gcount() - 1;
    if( readin < LINEMAX ) {
      buf[readin++] = '\n';	//  place newline back in string
      buf[readin] = '\0';       //  append a null character
    }

    istringstream iss(buf);
    
    while( c = iss.get() ) {  // remove whitespace and =
      if ( c != ' ' && c != '\t' && c != '\n' && c != '=')
	break;
    }
    
    iss.putback(c);
    while( iss.good()) {

      char *p = tok;
      while (c = iss.get()){
	if( c != ',' && c != '\n') {
	  *p = c; 
	  p++;
	}
	else break;
      }
      *p = '\0';		// Put names in tok
      
      token = tok;
      if(token == "Z")  TwoD = false;
      if (token != "" ){
	variables.add(token);
	setMaterialNum( token, nMaterials );
      }
      
      if ( c == '\n' ) {	// The last character is a newline
	break;			// we're done reading.
      }
      c = iss.peek();		// If we have made it here
      if( c == '\n' ){          // the previous char was a comma and
	c = ',';		//  now we have a newline.  We must 
        break;                  // read another line of variables.
      }         		
    }
    if ( c == '\n' ) break;     // we're done
  }
  int i;
  for(i = 0; i < variables.size(); i++)
    cout << variables[i] << " ";
  cout << endl << "nvars = "<< variables.size()<< endl;
  cout << "nMaterials = "<<nMaterials<<endl;

				//  Now create materials
  materials.remove_all();

  for(i = 0; i < nMaterials; i++)
    {
      materials.add( scinew MPMaterial() );
   }
	
}

void removeChars(istream& is)
{
  char c;
  while ( c = is.get() ) {
    if( c != '"' && c != ' ' && c != '=' && c != ',')
      break;
  }
}

int TecplotReader::find( char c, char *buf)
{
  istringstream iss(buf);
  char tok[LINEMAX];
  int i;
  iss.get(tok,LINEMAX,',');
  iss.get(); //remove comma
  if( iss.get(tok, LINEMAX, c) )
    {
      //      cerr<< tok << "\t";
      if( iss.get(tok, LINEMAX, '=') ) {
	iss.get(); // remove =
	
	if( iss.get(tok, LINEMAX, ',')) {
	  clString str(tok);
	  str.get_int(i);
	  //	  cerr << c <<" = "<<i<<endl;
	  return i;
	} else return 0;
      } else return 0;
    } else return 0;
}

int TecplotReader::isBlock( char *buf)
{
  if( find( 'I', buf ) && find('J', buf) )
    return 1;
  else
    return 0;
}
	 
  
  
void TecplotReader::readZone(istream& is)
{
  char *p;
  char buf[LINEMAX];
  char num[2];
  int i = 1, j = 1, k = 1, materialIndex = 1; // index is the material index
  
  if( is.getline(buf, LINEMAX) ){
    if( isBlock( buf ) )
      {
	i = find( 'I', buf );
	j = find( 'J', buf );
	k = find( 'K', buf );

        k = ( (k == 0) ? 1 : k);
	
	readBlock( i, j, k, is);
      } else {
	i = find( 'I', buf);
	p = buf;
	while ( *p != '"') p++;
	while ( !isdigit(*p) && *p != '"') p++;
	if( *p == '"' ){
	  //cerr<<"Error in Particle format\n";
	  num[0] = '1';
	} else {
	  num[0] = *p;
	}
	num[1] = '\0';
	materialIndex = atoi(num) -1;
	readParticles( i, materialIndex, is);
      }
   }
}

void TecplotReader::readParticles(int ii, int materialIndex, istream& is)
{
  // Each particle variable is read in order.  e.g. for particle 1
  // read all variables, for particle 2 read all variables, etc.
  // Unfortuately the cfdlibParticleSet stores variables in arrays
  // that correspond to a position array--- all positions are in 
  // one big array, all Ts are in one array, all UVWs, etc.
  // To add to the confusion a particle only corresponds to 1 material
  // even though the data file contains data points that correspond
  // to all materials.  Let's say that we have variables X,Y,P,T1,RO1,U1,V1
  // T2,RO2,U2,V2.  In this case every value beyond V1 is thrown away 
  // and should only contain zeros anyhow.  I have already determined
  // which material this particle corresponds to and have passed that info
  // in via the materialIndex variable.
  int i, v, index = 0;
  clString var, stripped;
  MPVizParticleSet *ps = scinew MPVizParticleSet();
  cfdlibTimeStep* ts = scinew cfdlibTimeStep();
  

  // First lets walk through the variables and add them to the 
  // particle set, ignoring any variables beyond the "1" vars.
  // Also we must index the variables that correspond to vectors. 
  // Generally this is position, direction and momentum. Finally
  // We keep an index to the last "1" variable so we know we can 
  // throw away anthing beyond that.
  Array1< int > vids;
  Array1< int > sids;
  int lastID = 0;
  for( v = 0; v < variables.size(); v++) {

    var = variables[v];
    const char *p = var();
    stripVar( var, stripped, index );
    if (index > 1 ) continue;  // we don't care
    else {
      if( *p == 'X'){
	vids.add(v);   // a vectorvar index
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "XY" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "XYZ" ) );
	}
      } else if( *p == 'U' ) {
	vids.add(v);  // a vectorvar index
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "UV" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "UVW" ) );
	}
      } else if( *p == 'A' ) {
	vids.add(v);  // a vectorvar index
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "AXY" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "AXYZ" ) );
	}
      } else if( *p == 'M' && *(++p) == 'O' ) {
	vids.add(v);  // a vectorvar index
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "MOXY" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "MOXYZ" ) );
	}
      } else if( *p == 'S' && *(++p) == 'N') {
	vids.add(v);
	if( TwoD ) {
	  v++;
	  ps->addVectorVar( clString( "SNXY" ) );
	} else {
	  v += 2;
	  ps->addVectorVar( clString( "SNXYZ" ) );
	}
      } else {   // a scalar variable
	ps->addScalarVar( stripped );
	sids.add(v);
      }
      lastID = v;  // get pointer to the last "good" var index.
    }
  }

  // Now we know where our vectors are, and the last good index. 
  // We also know how many scalar and vector variables we have.
  // Let's resize the timestep array.
  ts->vectors.resize( vids.size() );
  ts->scalars.resize( sids.size() );
 
  float x, y, z;

  for(i = 0 ; i < ii; i++ ){
    int vid = 0;  
    int sid = 0;

    for(v = 0; v < variables.size(); v++) {
      if ( v > lastID ){
	is >> x;
      } else {
	if (vid < vids.size() && v == vids[vid]){ // we have a vector variable
	  if( TwoD ) {
	    is >> x >> y;
	    ts->vectors[vid].add( Vector( x, y, 0) );
	    v++;
	  } else {
	    is >> x >> y >> z;
	    ts->vectors[vid].add( Vector( x, y, z) );
	    v+=2;
	  }
	  vid++; 
	} else {  // we have a scalar variable
	  is >> x;
	  ts->scalars[sid].add( x );
	  sid++;
	}
      }
    }
  }

  // Add the timestep to the particle set
  ps->add( ts );

  ParticleSetHandle psh = ParticleSetHandle( ps );
  materials[materialIndex]->AddParticleSet( psh );
}

void TecplotReader::readBlock(int i, int j, int k, istream& is)
{
  int index = 1;  // Note that for the ME guys indexing starts at 1.
  clString stripped;


  cerr<<"Working on Block of size "<< i << " by "<<j<<" by "<< k <<endl;

  for(int v = 0; v < variables.size(); v++)
    {
      clString var = variables[v];
      const char * p = var();
      //      cerr << var <<endl;
      if( var == "X"){ // || variables[v] == "Y") {
	getBounds(xmin, xmax, i, j, k, is);
      } else if(var == "Y") {
	getBounds(ymin, ymax, i, j, k, is);
      } else if(var == "Z") {
	getBounds(zmin, zmax, i, j, k, is);
      } else if(var == "P") {
	ScalarFieldHandle sfh = makeScalarField(i,j,k,is);
	for(int f=0; f < materials.size(); f++)
	  materials[f]->AddScalarField( var, sfh );
      }	else if ( *p == 'U' ){
				//  We can assume the next two (Possibly 
				//  three) variables make up the vectorfield
	VectorFieldHandle vfh = makeVectorField(i,j,k,is);
	stripVar(var,  stripped, index );
	if( TwoD ) {
	  v ++;
	  materials[index-1]->AddVectorField( clString("UV"), vfh);
	} else {
	  v += 2;
	  materials[index-1]->AddVectorField( clString("UVW"), vfh);
	}
      } else if ( *p == 'A' ){
	VectorFieldHandle vfh = makeVectorField(i,j,k,is);
	stripVar(var,  stripped, index );
	if( TwoD ) {
	  v ++;
	  materials[index-1]->AddVectorField( clString("AXY"), vfh);
	} else {
	  v += 2;
	  materials[index-1]->AddVectorField( clString("AXYZ"), vfh);
	}
      } else if ( *p == 'M' && *(++p) == 'O' ) { // Momentum
	VectorFieldHandle vfh = makeVectorField(i,j,k,is);
	stripVar(var,  stripped, index );
	if( TwoD ) {
	  v ++;
	  materials[index-1]->AddVectorField( clString("MOXY"), vfh);
	} else {
	  v += 2;
	  materials[index-1]->AddVectorField( clString("MOXYZ"), vfh);
	}
      } else if ( *p  == 'S' && *(++p) == 'N' ) { // Momentum
	VectorFieldHandle vfh = makeVectorField(i,j,k,is);
	stripVar(var,  stripped, index );
	if( TwoD ) {
	  v ++;
	  materials[index-1]->AddVectorField( clString("SNXY"), vfh);
	} else {
	  v += 2;
	  materials[index-1]->AddVectorField( clString("SNXYZ"), vfh);
	}
      } else {
	ScalarFieldHandle sfh = makeScalarField(i,j,k, is);
	stripVar(var, stripped, index);
	materials[index-1]->AddScalarField( stripped, sfh);
      }
    }      
  NOT_FINISHED("TecplotReader::readBlock");
}


void TecplotReader::getBounds(double& min, double& max,
			      int ii, int jj, int kk,
			      istream& is)
{
  int i,j,k;
  double val;
  min= 1.0E+50;
  max= -1.0E+50;
  for(k = 0; k < kk; k++)
    for(j = 0; j < jj; j++)
      for(i = 0; i < ii; i++){
	is >> val;
	max = ( max < val ) ? val : max;
	min = ( min > val ) ? val : min;
      }
}


void TecplotReader::setMaterialNum( clString str, int& nMaterials)
{
  for(int i = 2; i < MAXMATERIALS; i++){
    clString n(to_string(i));
    const char *p = n();
    if ( strchr(str(), p[0]) != NULL && nMaterials < i){
      nMaterials = i;
    }
  }
}


void TecplotReader::stripVar( const clString& var, clString& retVar, int& index )
{
  const char *p = var();
  char v[VARNAMELEN];
  int vindex = 0;
  for (int i = 0; i < var.len(); i++)
    {
      if (isdigit( *p )){
	char num[2];
	num[0] = *p;
	num[1] = '\0';
	index = atoi( num );
	p++;
      }
      else {
	v[vindex++] = *p;
	p++;
      }
    }
  v[vindex] = '\0';
  retVar = v;
}
  
// ScalarFieldHandle TecplotReader::makeScalarField(int, int, int, istream&)
ScalarFieldHandle TecplotReader::makeScalarField(int ii, int jj,
						 int kk, istream& is)
{

  ScalarFieldRG *sf = scinew ScalarFieldRG();
  int i, j, k;
  float s;

  if( TwoD ) {  // zmin = zmax.  We must make the field 3D.
    sf->resize(ii,jj,2);
    sf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax+1));
  } else {
    sf->resize(ii,jj,kk);
    sf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax));
  }

  for(k = 0; k < kk; k++){
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	is >> s;
	sf->grid(i,j,k) = s;
	if( TwoD ){
	  sf->grid(i,j,k+1) = s;
	}
      }
    }
  }
  return ScalarFieldHandle( sf );
}

//VectorFieldHandle TecplotReader::makeVectorField(int,int,int,istream&)
VectorFieldHandle TecplotReader::makeVectorField(int ii, int jj,
						 int kk, istream& is)
{
  VectorFieldRG *vf = scinew VectorFieldRG();
  
  int i,j,k;
  float x,y,z;

  if( TwoD ) {  // zmin = zmax.  We must make the field 3D.
    vf->resize(ii,jj,2);
    vf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax+1));
  } else {
    vf->resize(ii,jj,kk);
    vf->set_bounds(Point(xmin,ymin,zmin), Point(xmax,ymax,zmax));
  }
    

  for(k = 0; k < kk; k++){  //read all the the first var
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	is >> x;
	vf->grid(i,j,k) = Vector(x,0,0);
      }
    }
  }

  for(k = 0; k < kk; k++){ // read the second var
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	is >> y;  
	vf->grid(i,j,k).y(y);
      }
    }
  }

  if ( TwoD ) { // no more reading
    for(j = 0; j < jj; j++){
      for(i = 0; i < ii; i++ ) {
	vf->grid(i,j,1) = vf->grid(i,j,0);
      }
    }
  } else {
    for(k = 0; k < kk; k++){
      for(j = 0; j < jj; j++){
	for(i = 0; i < ii; i++ ) {
	  is >> z;
	  vf->grid(i,j,k).z(z);
	}
      }
    }
  }
  return VectorFieldHandle( vf );
}

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.7  1999/12/28 21:11:45  kuzimmer
// modified so that picking works again
//
// Revision 1.6  1999/10/07 02:08:28  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/09/21 16:12:25  kuzimmer
// changes made to support binary/ASCII file IO
//
// Revision 1.4  1999/08/25 03:49:04  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.3  1999/08/18 21:45:27  sparker
// Array1 const correctness, and subsequent fixes
// Array1 bug fix courtesy Tom Thompson
//
// Revision 1.2  1999/08/17 06:40:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 17:08:58  mcq
// Initial commit
//
// Revision 1.4  1999/07/07 21:11:10  dav
// added beginnings of support for g++ compilation
//
// Revision 1.3  1999/06/09 23:23:44  kuzimmer
// Modified the modules to work with the new Material/Particle classes.  When a module needs to determine the type of particleSet that is incoming, the new stl dynamic type testing is used.  Works good so far.
//
// Revision 1.2  1999/04/27 23:18:41  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
