
/*
 *  mpmParticleSet.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Modules/CD/mpmParticleSet.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew mpmParticleSet();
}

PersistentTypeID mpmParticleSet::type_id("mpmParticleSet", "ParticleSet", maker);

mpmParticleSet::mpmParticleSet()
{
}

mpmParticleSet::~mpmParticleSet()
{
    for(int i=0;i<timesteps.size();i++)
	delete timesteps[i];
}

void mpmParticleSet::add(mpmTimeStep* ts)
{
    timesteps.add(ts);
}

int mpmParticleSet::position_vector()
{
    return 0;
}

void mpmParticleSet::get(int timestep, int,
			    Array1<Vector>& values, int start, int end)
{
    mpmTimeStep* ts=timesteps[timestep];
    if(start==-1)
	start=0;
    if(end==-1)
	end=ts->positions.size();
    int n=end-start;
    values.resize(n);
    for(int i=0;i<n;i++)
	values[i]=ts->positions[i+start];
}

void mpmParticleSet:: list_natural_times(Array1<double>& times)
{
    times.resize(timesteps.size());
    for(int i=0;i<timesteps.size();i++)
	times[i]=timesteps[i]->time;
}

#define mpmPARTICLESET_VERSION 1

void mpmParticleSet::io(Piostream& stream)
{
    stream.begin_class("mpmParticleSet", mpmPARTICLESET_VERSION);
    ParticleSet::io(stream);
    int nsets=timesteps.size();
    Pio(stream, nsets);
    if(stream.reading())
	timesteps.resize(nsets);
    for(int i=0;i<nsets;i++){
	if(stream.reading())
	    timesteps[i]=new mpmTimeStep();
	Pio(stream, timesteps[i]->time);
	Pio(stream, timesteps[i]->positions);
    }
    stream.end_class();
}

// The following functions added by psutton to make things compile.
// I doubt any do what they're supposed to.
ParticleSet *mpmParticleSet::clone() const
{
  return scinew mpmParticleSet();
}

int mpmParticleSet::find_scalar(const clString&)
{
  return 0;
}

void mpmParticleSet::list_scalars(Array1<clString>&)
{
}

int mpmParticleSet::find_vector(const clString&)
{
  return 0;
}

void mpmParticleSet::list_vectors(Array1<clString>&)
{
}
  
void mpmParticleSet::get(int timestep, int, Array1<double>& value,
			    int start, int end)
{
  mpmTimeStep* ts=timesteps[timestep];
  if(start==-1)
    start=0;
  if(end==-1)
    end=ts->positions.size();
  int n=end-start;
  value.resize(n);
  for(int i=0;i<n;i++)
    value[i]=ts->scalars[i+start];
}

void mpmParticleSet::interpolate(double, int, Vector&,
				    int, int)
{
}

void mpmParticleSet::interpolate(double, int, double&,
				    int, int)
{
}

void mpmParticleSet::print() {
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
