
/*
 *  cfdlibParticleSet.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Datatypes/cfdlibParticleSet.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

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

int cfdlibParticleSet::position_vector()
{
    return 0;
}

void cfdlibParticleSet::get(int timestep, int vectorid,
			    Array1<Vector>& values, int start, int end)
{
    cfdlibTimeStep* ts=timesteps[timestep];
    if(start==-1)
	start=0;
    if(end==-1)
	end=ts->positions.size();
    int n=end-start;
    values.resize(n);
    for(int i=0;i<n;i++)
	values[i]=ts->positions[i+start];
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
    stream.begin_class("cfdlibParticleSet", cfdlibPARTICLESET_VERSION);
    ParticleSet::io(stream);
    int nsets=timesteps.size();
    Pio(stream, nsets);
    if(stream.reading())
	timesteps.resize(nsets);
    for(int i=0;i<nsets;i++){
	if(stream.reading())
	    timesteps[i]=new cfdlibTimeStep();
	Pio(stream, timesteps[i]->time);
	Pio(stream, timesteps[i]->positions);
    }
    stream.end_class();
}

// The following functions added by psutton to make things compile.
// I doubt any do what they're supposed to.
ParticleSet *cfdlibParticleSet::clone() const
{
  return scinew cfdlibParticleSet();
}

int cfdlibParticleSet::find_scalar(const clString& name)
{
  return 0;
}

void cfdlibParticleSet::list_scalars(Array1<clString>& names)
{
}

int cfdlibParticleSet::find_vector(const clString& name)
{
  return 0;
}

void cfdlibParticleSet::list_vectors(Array1<clString>& names)
{
}
  
void cfdlibParticleSet::get(int timestep, int scalarid, Array1<double>& value,
			    int start, int end)
{
  cfdlibTimeStep* ts=timesteps[timestep];
  if(start==-1)
    start=0;
  if(end==-1)
    end=ts->positions.size();
  int n=end-start;
  value.resize(n);
  for(int i=0;i<n;i++)
    value[i]=ts->scalars[i+start];
}

void cfdlibParticleSet::interpolate(double time, int vectorid, Vector& value,
				    int start, int end)
{
}

void cfdlibParticleSet::interpolate(double time, int scalarid, double& value,
				    int start, int end)
{
}

void cfdlibParticleSet::print() {
  cout << "Particle Set.  t = " << timesteps[0]->time << endl;
  int i;
  int n = timesteps[0]->positions.size();
  for(i=0;i<n;i++) {
    cout << (timesteps[0]->positions[i]).x() << " " 
	 << (timesteps[0]->positions[i]).y() << " "
	 << (timesteps[0]->positions[i]).z() << ":\t"
	 << timesteps[0]->scalars[i] << endl;
  }
}
