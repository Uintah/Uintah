
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

void cfdlibParticleSet::get(int timestep, int vectorid, Array1<Vector>& values,
			    int start, int end)
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


