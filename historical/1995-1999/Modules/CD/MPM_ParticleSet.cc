
/*
 *  MPM_ParticleSet.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Datatypes/MPM_ParticleSet.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew MPM_ParticleSet();
}

PersistentTypeID MPM_ParticleSet::type_id("MPM_ParticleSet", "ParticleSet", maker);

MPM_ParticleSet::MPM_ParticleSet()
{
}

MPM_ParticleSet::~MPM_ParticleSet()
{
    for(int i=0;i<timesteps.size();i++)
	delete timesteps[i];
}

void MPM_ParticleSet::add(MPM_TimeStep* ts)
{
    timesteps.add(ts);
}

int MPM_ParticleSet::position_vector()
{
    return 0;
}

void MPM_ParticleSet::get(int timestep, int,
			    Array1<Vector>& values, int start, int end)
{
    MPM_TimeStep* ts=timesteps[timestep];
    if(start==-1)
	start=0;
    if(end==-1)
	end=ts->positions.size();
    int n=end-start;
    values.resize(n);
    for(int i=0;i<n;i++)
	values[i]=ts->positions[i+start];
}

void MPM_ParticleSet:: list_natural_times(Array1<double>& times)
{
    times.resize(timesteps.size());
    for(int i=0;i<timesteps.size();i++)
	times[i]=timesteps[i]->time;
}

#define MPM_PARTICLESET_VERSION 1

void MPM_ParticleSet::io(Piostream& stream)
{
    stream.begin_class("MPM_ParticleSet", MPM_PARTICLESET_VERSION);
    ParticleSet::io(stream);
    int nsets=timesteps.size();
    Pio(stream, nsets);
    if(stream.reading())
	timesteps.resize(nsets);
    for(int i=0;i<nsets;i++){
	if(stream.reading())
	    timesteps[i]=new MPM_TimeStep();
	Pio(stream, timesteps[i]->time);
	Pio(stream, timesteps[i]->positions);
    }
    stream.end_class();
}

// The following functions added by psutton to make things compile.
// I doubt any do what they're supposed to.
ParticleSet *MPM_ParticleSet::clone() const
{
  return scinew MPM_ParticleSet();
}

int MPM_ParticleSet::find_scalar(const clString&)
{
  return 0;
}

void MPM_ParticleSet::list_scalars(Array1<clString>&)
{
}

int MPM_ParticleSet::find_vector(const clString&)
{
  return 0;
}

void MPM_ParticleSet::list_vectors(Array1<clString>&)
{
}
  
void MPM_ParticleSet::get(int timestep, int, Array1<double>& value,
			    int start, int end)
{
  MPM_TimeStep* ts=timesteps[timestep];
  if(start==-1)
    start=0;
  if(end==-1)
    end=ts->positions.size();
  int n=end-start;
  value.resize(n);
  for(int i=0;i<n;i++)
    value[i]=ts->scalars[i+start];
}

void MPM_ParticleSet::interpolate(double, int, Vector&,
				    int, int)
{
}

void MPM_ParticleSet::interpolate(double, int, double&,
				    int, int)
{
}

void MPM_ParticleSet::print() {
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
