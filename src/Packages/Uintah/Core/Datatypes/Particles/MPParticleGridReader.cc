//static char *id="@(#) $Id$";

/*
 *  MPParticleGridReader.cc: 
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 1999 SCI Group
 */
#include "MPParticleGridReader.h"
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>

#include <fstream>
#include <iomanip>
#include <sstream>
using namespace std;

namespace Uintah {
namespace Datatypes {

using SCICore::Containers::basename;
using SCICore::Containers::pathname;
using SCICore::Containers::to_string;

static Persistent* maker()
{
    return new MPParticleGridReader();
}

PersistentTypeID MPParticleGridReader::type_id("MPParticleGridReader",
					"ParticleGridReader", maker);

MPParticleGridReader::~MPParticleGridReader()
{
}

MPParticleGridReader::MPParticleGridReader() : xmin(0), xmax(1), ymin(1), ymax(1),
  zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0)
{
}

MPParticleGridReader::MPParticleGridReader(const clString& filename, int start,
			     int end, int incr) : xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime( start ),
  endTime( end ), increment(incr), filename(filename)
{
  readfile();
}

MPParticleGridReader::MPParticleGridReader(const clString& file ) : xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0), filename(file)
{
  readfile();
}

void MPParticleGridReader::SetFile(const clString& file )
{
  filename = file;
}

clString MPParticleGridReader::GetFile()
{
  return filename;
}

int MPParticleGridReader::GetNTimesteps()
{
  return endTime - startTime;
}

void MPParticleGridReader::GetParticleData(int particleId,
			       clString pSetName,
			       clString varname,
			       Array1<double>& values)
{
  if(  endTime - startTime <= 0 )
    return;

  values.remove_all();
  
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

  // before we begin we must determine the variable index in the MPParticleGrid 
  // file.  variableId gives us the index to the nth vector or scalar.
  // by checking isVector we know to which variableId refers.

  for( ii = startTime; ii <= endTime; ii += increment){
    ostringstream ostr;
    ostr.fill('0');
    ostr << path << "/"<< root<< setw(4)<<ii;
    
    MPRead read( ostr.str().c_str() );
    cerr<< "filename is "<< ostr.str().c_str() << endl;
    double value;
    read.GetParticleVariableValue( particleId, pSetName, varname, value );
    values.add( value );
  }

  NOT_FINISHED("MPParticleGridReader::GetParticleData");
}
void MPParticleGridReader::GetParticleData(int particleId,
			      int variableId,
			      int materialId,
			      bool isVector,
					   Array1<float>& values)
{
  NOT_FINISHED("MPParticleGridReader::GetParticleData");
}

ParticleSetHandle MPParticleGridReader::GetParticleSet( clString name )
{
    std::cerr<<"Trying to obtain particleset "<<name<<endl;
  map<clString, ParticleSetHandle, ltstr>::iterator it = psmap.find( name );
  if (it == psmap.end())
    return 0;
  else
    return psmap[ name ];
}     

VizGridHandle MPParticleGridReader::GetGrid( clString name )
{
    std::cerr<< "Trying to obtain grid "<<name<<endl;
  map<clString, VizGridHandle, ltstr>::iterator it = vgmap.find( name );
  if (it == vgmap.end())
    return 0;
  else
    return vgmap[ name ];
}
 
ParticleSetHandle MPParticleGridReader::GetParticleSet(int i )
{
  
  map<clString, ParticleSetHandle, ltstr>::iterator it = psmap.begin();
  for(int j = 0; j < i; j++, it++);
  if (it == psmap.end())
    return 0;
  else
    return (*it).second;
}     
VizGridHandle MPParticleGridReader::GetGrid( int i )
{
  map<clString, VizGridHandle, ltstr>::iterator it = vgmap.begin();
  for(int j = 0; j < i; j++, it++);
  if (it == vgmap.end())
    return 0;
  else
    return (*it).second;
}    


#define MPPARTICLEGRIDREADER_VERSION 1

void MPParticleGridReader::io(Piostream& stream)
{
  stream.begin_class("MPParticleGridReader", MPPARTICLEGRIDREADER_VERSION);
  ParticleGridReader::io(stream);

  stream.end_class();
}


int MPParticleGridReader::readfile()
{
  //NOT_FINISHED("MPParticleGridReader::readfile");

  MPRead  reader( filename );

  clString title, comments, datatype, fileType;
  Array1<clString> m;

  while(reader.ReadBlock( datatype )){
    if( datatype == "GRID"){
      readGrid( reader );
    } else if ( datatype == "PARTICLES" ){
      readParticles( reader );
    } else if ( datatype == "HEADER" ) {
      reader.ReadHeader( title, fileType, comments );
    } else {
      cerr<<"Error:  unknown Block type.\n";
      return 0;
    }
  }
  return 1;
}

void MPParticleGridReader::readGrid( MPRead& reader)
{
  clString name, type;
  Array1< double > X,Y,Z;
  int x,y,z;
  int i;
  Array1<clString> sVars;
  Array1<clString> vVars;
  clString varname;
  clString scalarVars, vectorVars;
  clString mat;
  double o_x, o_y, o_z, dx, dy, dz;
  o_x = o_y = o_z = dx = dy = dz = 0;

  reader.GetGridInfo( name, type, x,y,z, sVars, vVars);
  MPVizGrid *grid = new MPVizGrid(name);

  if( type == "NC" || type == "CC" ){
    reader.GetGridPoints(o_x, o_y, o_z, dx, dy, dz);
    reader.getScalarVars( sVars );
    for(i = 0; i < sVars.size(); i++){
      ScalarFieldHandle sfh;
      reader.GetScalarField( sfh );
      grid->AddScalarField(sVars[i], sfh );
    }
    reader.getVectorVars( vVars );
    for(i = 0; i < vVars.size(); i++){
      VectorFieldHandle vfh;
      reader.GetVectorField( vfh );
      grid->AddVectorField(vVars[i], vfh );
    }
    std::cerr <<"Adding grid "<< name << " to vgmap "<< std::endl;
    vgmap[ name ] = VizGridHandle( grid );
    
  }
}
      


void MPParticleGridReader::readParticles(MPRead& reader)
{
  int i,j;
  clString name;
  Array1<clString> sVars;
  Array1<clString> vVars;
  int nParticles;
  cfdlibTimeStep* ts = new cfdlibTimeStep();

  reader.GetParticleInfo(name, nParticles, sVars, vVars);
  MPVizParticleSet *ps = new MPVizParticleSet(name);
  
  for( i = 0; i < sVars.size(); i++){
    ps->addScalarVar( sVars[i] );
  }
  ps->addVectorVar( "XYZ" );
  for (i = 0; i< vVars.size(); i++){
    ps->addVectorVar( vVars[i] );
  }
  
  ts->vectors.resize( vVars.size() +1 );
  ts->scalars.resize( sVars.size() );
  
  Point p;
  Array1< double > scalars;
  Array1< Vector > vectors;
  for( i = 0; i < nParticles; i++ ){
    reader.GetParticle(p, scalars, vectors);
    ts->vectors[0].add( p.vector() );
    for(j = 0; j < scalars.size(); j++ ){
      ts->scalars[j].add( scalars[j] );
    }
    for( j = 0; j < vectors.size(); j++ ){
      ts->vectors[j+1].add( vectors[j] );
    }
  }

  ps->add(ts);
  ParticleSetHandle psh = ParticleSetHandle( ps );
  std::cerr <<   "adding particleset "<< name << " to psmap"<< std::endl;
  psmap[name] = psh;

}



} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.4  1999/12/28 21:09:08  kuzimmer
// modified file readers so that we can read multiple files for parallel output
//
// Revision 1.3  1999/11/18 22:13:43  jsday
// fixed Uintah and DaveW stuff to use STL iostream
//
// Revision 1.2  1999/10/07 02:08:23  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/21 16:08:29  kuzimmer
// modifications for binary file format
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
