
/*
 *  MFMPParticleGridReader.cc: 
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 1999 SCI Group
 */
#include "MFMPParticleGridReader.h"
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>

#include <fstream>
//#include <ostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algo.h>
using std::endl;
using std::cerr;
using std::string;
using std::vector;
using std::min;
using std::max;
using std::istringstream;

namespace Uintah {
namespace Datatypes {

using SCICore::Containers::basename;
using SCICore::Containers::pathname;
using SCICore::Containers::to_string;

static Persistent* maker()
{
    return scinew MFMPParticleGridReader();
}

PersistentTypeID MFMPParticleGridReader::type_id("MFMPParticleGridReader",
					"ParticleGridReader", maker);

MFMPParticleGridReader::~MFMPParticleGridReader()
{
}

MFMPParticleGridReader::MFMPParticleGridReader() : 
  xmin(0), xmax(1), ymin(1), ymax(1),
  zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0)
{
}

MFMPParticleGridReader::MFMPParticleGridReader(const clString& files,
					       int start, int end, int incr) :
  xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime( start ),
  endTime( end ), increment(incr), filenames(files)
{
  readfile();
}

MFMPParticleGridReader::MFMPParticleGridReader(const clString& files) :
  xmin(0), xmax(1),
  ymin(1), ymax(1), zmin(0), zmax(1), TwoD( true ), startTime(0),
  endTime(0), increment(0), filenames(files)
{
  readfile();
}

void MFMPParticleGridReader::SetFile(const clString& files )
{
  filenames = files;
}

clString MFMPParticleGridReader::GetFile()
{
  return filenames;
}

int MFMPParticleGridReader::GetNTimesteps()
{
  return endTime - startTime;
}

void MFMPParticleGridReader::GetParticleData(int particleId,
					     clString pSetName,
					     clString varname,
					     Array1<double>& values)
{
  NOT_FINISHED("MFMPParticleGridReader::GetParticleData");
}
void MFMPParticleGridReader::GetParticleData(int particleId,
					     int variableId,
					     int materialId,
					     bool isVector,
					     Array1<float>& values)
{
  NOT_FINISHED("MFMPParticleGridReader::GetParticleData");
}

ParticleSetHandle MFMPParticleGridReader::GetParticleSet( clString name )
{
  std::cerr<<"Trying to obtain particleset "<<name<<endl;
  map<clString, ParticleSetHandle, ltstr>::iterator it = psmap.find( name );
  if (it == psmap.end())
    return 0;
  else
    return psmap[ name ];
}     

VizGridHandle MFMPParticleGridReader::GetGrid( clString name )
{
  std::cerr<< "Trying to obtain grid "<<name<<endl;
  map<clString, VizGridHandle, ltstr>::iterator it = vgmap.find( name );
  if (it == vgmap.end())
    return 0;
  else
    return vgmap[ name ];
}
 
ParticleSetHandle MFMPParticleGridReader::GetParticleSet(int i )
{
  
  map<clString, ParticleSetHandle, ltstr>::iterator it = psmap.begin();
  for(int j = 0; j < i; j++, it++);
  if (it == psmap.end())
    return 0;
  else
    return (*it).second;
}     
VizGridHandle MFMPParticleGridReader::GetGrid( int i )
{
  map<clString, VizGridHandle, ltstr>::iterator it = vgmap.begin();
  for(int j = 0; j < i; j++, it++);
  if (it == vgmap.end())
    return 0;
  else
    return (*it).second;
}    


#define MPPARTICLEGRIDREADER_VERSION 1

void MFMPParticleGridReader::io(Piostream& stream)
{
  stream.begin_class("MFMPParticleGridReader", MPPARTICLEGRIDREADER_VERSION);
  ParticleGridReader::io(stream);

  stream.end_class();
}


int MFMPParticleGridReader::readfile()
{
  //NOT_FINISHED("MFMPParticleGridReader::readfile");

  // put multiple file reading here.


  istringstream is( filenames());
  vector<string> fnames;
  string in;
  while ( is >> in ){
    cerr << in << endl;
    fnames.insert(fnames.end(), in);
  }
  
  vector<MPRead *> readers;
  for( int i = 0; i < (int)fnames.size(); i++){
    readers.insert(readers.end(), scinew MPRead( fnames[i].c_str() ));
  }
  
  clString title;
  clString filetype;
  clString comments;
  
  for(int i = 0; i < (int)readers.size(); i++) {
    readers[i]->ReadHeader( title, filetype, comments);
  }
  
  clString blocktype;
  int returnval = 1;
  while ( returnval ) {
    for(int i = 0; i < (int)readers.size(); i++) {
      returnval = readers[i]->ReadBlock( blocktype );
    }
    
    if (blocktype == "GRID" ){
      readGrid( readers );
    } else if ( blocktype == "PARTICLES" ){
      readParticles( readers );
    } else {
      cerr<<"Error:  unknown Block type: "<< blocktype <<endl;
      return 0;
    }
  }
  return 1;
}

void MFMPParticleGridReader::readGrid( vector<MPRead*>& readers)
{

  clString name;
  clString type;
  int sizex, sizey, sizez;
  double ox, oy, oz;
  double dx,dy,dz;
  Array1<clString> scalars;
  Array1<clString> vectors;

  double xmin,ymin,zmin, xmax,ymax,zmax;
  xmin = ymin = zmin = 1.0e6;
  xmax = ymax = zmax = -1.0e6;
  for(int i = 0; i < (int)readers.size(); i++) {
    readers[i]->GetGridInfo( name , type, sizex, sizey, sizez,
                             scalars, vectors);
    readers[i]->GetGridPoints( ox, oy, oz, dx, dy, dz );
    xmin = min( xmin, ox );
    ymin = min( ymin, oy );
    zmin = min( zmin, oz );
    xmax = max ( xmax, double(ox + (sizex * dx)));
    ymax = max ( ymax, double(oy + (sizey * dy)));
    zmax = max ( zmax, double(oz + (sizez * dz)));
  }

  sizex = (xmax - xmin)/dx;
  sizey = (ymax - ymin)/dy;
  sizez = (zmax - zmin)/dz;


  clString svars = "";
  clString vvars = "";
  for(int i = 0; i < (int)scalars.size(); i++)
    svars = svars+scalars[i]+" ";
  for(int i = 0; i < (int)vectors.size(); i++)
    vvars = vvars+vectors[i]+" ";

  // create the single scalar field
  MPVizGrid *grid = scinew MPVizGrid(name);

  Point minPt(xmin,ymin,zmin);
  Point maxPt(xmax, ymax, zmax);
  Point dPt(dx,dy,dz);
  ScalarFieldHandle sfh;
  int si, sj, sk;
  int i,j,k;
  for(int ss = 0; ss< (int)scalars.size(); ss++){
    ScalarFieldRG *sfrg = scinew ScalarFieldRG();
    sfrg->set_bounds( minPt, maxPt );
    sfrg->resize( sizex, sizey, sizez );
    for(int rs = 0; rs < (int)readers.size(); rs++){
      readers[rs]->GetScalarField( sfh );
      ScalarFieldRG* sf = sfh->getRG();
      Point minp;
      Point maxp;
      sf->get_bounds( minp, maxp);
      getindices(minPt, minp, dPt, si, sj, sk);
      for(int ii = 0, i = si; ii < sf->nx; ii++, i++){
        for(int jj = 0, j = sj; jj < sf->ny; jj++, j++){
          for(int kk = 0, k = sk; kk < sf->nz; kk++, k++) {
            sfrg->grid(i,j,k) = sf->grid(ii,jj,kk);
          }
        }
      }
    }
    int pre = 0, post = 0;
    cerr<< "Adding Scalar variable "<<scalars[ss]<<" to grid "
	<< grid->getName() << endl;
    grid->AddScalarField(scalars[ss], ScalarFieldHandle(sfrg) );
    //    delete sfrg;
  }

  VectorFieldHandle vfh;
  for(int vs = 0; vs< (int)vectors.size(); vs++){
    VectorFieldRG *vfrg = scinew VectorFieldRG();
    vfrg->resize( sizex, sizey, sizez );
    vfrg->set_bounds( minPt, maxPt );

    for(int rs = 0; rs < (int)readers.size(); rs++){
      readers[rs]->GetVectorField( vfh );
      VectorFieldRG* vf = vfh->getRG();
      Point minp;
      Point maxp;
      vf->get_bounds( minp, maxp);
      getindices(minPt, minp, dPt, si, sj, sk);
      for(int ii = 0, i = si; ii < vf->nx; ii++, i++){
        for( int jj = 0, j = sj; jj < vf->ny; jj++, j++){
          for(int kk = 0, k= sk; kk < vf->nz; kk++, k++) {
            vfrg->grid(i,j,k) = vf->grid(ii,jj,kk);
          }
        }
      }
    }
    cerr<< "Adding Vector variable "<<vectors[vs]<<" to grid "
	<< grid->getName() << endl;
    grid->AddVectorField(vectors[vs], VectorFieldHandle(vfrg) );
    //    delete vfrg;
  }

  vgmap[ name ] = VizGridHandle( grid );
  cerr<<"So far so good \n";
}


void MFMPParticleGridReader::readParticles(vector<MPRead*>& readers)
{
  int i,j,k;
  clString name;
  Array1<clString> sVars;
  Array1<clString> vVars;
  Array1<int> nps;
  int total = 0, nParticles;
  cfdlibTimeStep* ts = scinew cfdlibTimeStep();
  for(i = 0; i < (int)readers.size(); i++) {
    readers[i]->GetParticleInfo(name, nParticles, sVars, vVars);
    nps.add( nParticles );
    total += nParticles;
  }
  MPVizParticleSet *ps = scinew MPVizParticleSet(name);
  
  for( i = 0; i < (int)sVars.size(); i++){
    ps->addScalarVar( sVars[i] );
  }
  ps->addVectorVar( "XYZ" );
  for (i = 0; i< (int)vVars.size(); i++){
    ps->addVectorVar( vVars[i] );
  }
  
  ts->vectors.resize( vVars.size() +1 );
  ts->scalars.resize( sVars.size() );
  
  Point p;
  Array1< double > scalars;
  Array1< Vector > vectors;
  for( k = 0; k < (int)readers.size(); k++){
    nParticles = nps[k];
    MPRead *reader = readers[k];
    for( i = 0; i < nParticles; i++ ){
      reader->GetParticle(p, scalars, vectors);
      ts->vectors[0].add( p.vector() );
      for(j = 0; j < (int)scalars.size(); j++ ){
	ts->scalars[j].add( scalars[j] );
      }
      for( j = 0; j < (int)vectors.size(); j++ ){
	ts->vectors[j+1].add( vectors[j] );
      }
    }
  }
  ps->add(ts);
  ParticleSetHandle psh = ParticleSetHandle( ps );
  std::cerr <<   "adding particleset "<< name << " to psmap"<< std::endl;
  psmap[name] = psh;

}

//-------------------------------------------------------------- 
void
MFMPParticleGridReader::getindices(Point minPt, Point subminPt, Point dPt,
			      int& i, int& j, int& k)
{
  i = (subminPt.x() -minPt.x())/dPt.x();
  j = (subminPt.y() -minPt.y())/dPt.y();
  k = (subminPt.z() -minPt.z())/dPt.z();
}


} // End namespace Datatypes
} // End namespace Uintah

