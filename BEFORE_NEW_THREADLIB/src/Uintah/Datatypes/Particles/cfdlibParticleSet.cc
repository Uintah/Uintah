//static char *id="@(#) $Id$";

/*
 *  cfdlibParticleSet.cc:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <iostream.h>

#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

#include <Uintah/Datatypes/Particles/cfdlibParticleSet.h>

namespace Uintah {
namespace Datatypes {

static Persistent* maker()
{
    return scinew cfdlibParticleSet();
}

PersistentTypeID cfdlibParticleSet::type_id("cfdlibParticleSet", "ParticleSet", maker);

cfdlibParticleSet::cfdlibParticleSet()
{
}

cfdlibParticleSet::~cfdlibParticleSet()
{
    for(int i=0;i<timesteps.size();i++)
	delete timesteps[i];
}

void cfdlibParticleSet::add(cfdlibTimeStep* ts)
{
    timesteps.add(ts);
}
void cfdlibParticleSet::addVectorVar(const clString& var)
{
  vectorvars.add(var);
}

void cfdlibParticleSet::addScalarVar(const clString& var)
{
  scalarvars.add(var);
}

int cfdlibParticleSet::position_vector()
{
    return 0;
}

void cfdlibParticleSet::get(int timestep, int index,
			    Array1<Vector>& values, int start, int end)
{
    cfdlibTimeStep* ts=timesteps[timestep];
    if(start==-1)
	start=0;
    if(end==-1)
	end=ts->vectors[index].size();
        // end=ts->positions.size();
    int n=end-start;
    values.resize(n);
    for(int i=0;i<n;i++)
        // values[i]=ts->positions[i+start];
      values[i] = (ts->vectors[index])[i + start];
}

void cfdlibParticleSet::get(int timestep, int index, Array1<double>& values,
			    int start, int end)
{
  cfdlibTimeStep* ts=timesteps[timestep];
  if(start==-1)
    start=0;
  if(end==-1)
    //end=ts->positions.size();
    end=ts->scalars[index].size();
  int n=end-start;
  values.resize(n);
  for(int i=0;i<n;i++)
    //value[i]=ts->scalars[i+start];
    values[i] = (ts->scalars[index])[i + start];
}

Vector cfdlibParticleSet::getVector(int timestep, int vectorid, int index)
{
  cfdlibTimeStep* ts=timesteps[timestep];
  return (ts->vectors[vectorid])[index];
}
double cfdlibParticleSet::getScalar(int timestep, int scalarid, int index)
{
  cfdlibTimeStep* ts=timesteps[timestep];
  return (ts->scalars[scalarid])[index];
}




void cfdlibParticleSet:: list_natural_times(Array1<double>& times)
{
    times.resize(timesteps.size());
    for(int i=0;i<timesteps.size();i++)
	times[i]=timesteps[i]->time;
}

#define cfdlibPARTICLESET_VERSION 1

void cfdlibParticleSet::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;

    stream.begin_class("cfdlibParticleSet", cfdlibPARTICLESET_VERSION);
    ParticleSet::io(stream);
    int nsets=timesteps.size();
    Pio(stream, nsets);
    if(stream.reading())
      timesteps.resize(nsets);
//     for(int i=0;i<nsets;i++){
// 	if(stream.reading())
// 	    timesteps[i]=new cfdlibTimeStep();
// 	Pio(stream, timesteps[i]->time);
// 	Pio(stream, timesteps[i]->positions);
//     }
    for(int i = 0; i < nsets; i++){
      if(stream.reading())
	timesteps[i] = new cfdlibTimeStep();
      Pio(stream, timesteps[i]->vectors);
      Pio(stream, timesteps[i]->scalars);
    }
    stream.end_class();
}

// The following functions added by psutton to make things compile.
// I doubt any do what they're supposed to.
ParticleSet *cfdlibParticleSet::clone() const
{
  return scinew cfdlibParticleSet();
}

int cfdlibParticleSet::find_scalar(const clString& var)
{
  for(int i = 0; i < scalarvars.size(); i++){
    if( var == scalarvars[i] )
      return i;
  }
  return -1;
}

void cfdlibParticleSet::list_scalars(Array1<clString>& svs)
{
  for(int i = 0; i < scalarvars.size(); i++)
    svs.add( scalarvars[i] );
}

int cfdlibParticleSet::find_vector(const clString& var)
{
  for(int i = 0; i < vectorvars.size(); i++){
    if( var == vectorvars[i] )
      return i;
  }
  return -1;

}

void cfdlibParticleSet::list_vectors(Array1<clString>&  vvs)
{
  for(int i = 0; i < vectorvars.size(); i++)
    vvs.add( vectorvars[i] );
}
  
void cfdlibParticleSet::interpolate(double, int, Vector&,
				    int, int)
{
}

void cfdlibParticleSet::interpolate(double, int, double&,
				    int, int)
{
}

void cfdlibParticleSet::print() {
  cout << "Particle Set.  t = " << timesteps[0]->time << endl;
  int i;
  int n = (timesteps[0]->vectors[0]).size();
  for(i=0;i<n;i++) {
    cout << ((timesteps[0]->vectors[0])[i]).x() << " " 
	 << ((timesteps[0]->vectors[0])[i]).y() << " "
	 << ((timesteps[0]->vectors[0])[i]).z() << ":\t"
	 << (timesteps[0]->scalars[0])[i] << endl;
  }
}

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.2  1999/08/17 06:40:09  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:59:01  mcq
// Initial commit
//
// Revision 1.2  1999/07/07 21:11:08  dav
// added beginnings of support for g++ compilation
//
// Revision 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//
