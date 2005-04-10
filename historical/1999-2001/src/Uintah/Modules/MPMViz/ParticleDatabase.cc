
/*
 *  ParticleDatabase: Simple multi-threaded particle database
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Uintah/Modules/MPMViz/ParticleDatabase.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Thread/Guard.h>
using SCICore::Thread::Guard;
#include <SCICore/Thread/Mutex.h>
using SCICore::Thread::Mutex;
#include <algorithm>
using std::find;
using std::for_each;
using std::generate;
using std::sort;
using std::swap;
#include <iostream>
using std::cerr;

using std::map;

using Uintah::Modules::ParticleDatabase;
using Uintah::Datatypes::Particles::TimestepNotify;
using std::vector;
using CIA::array2;

ParticleDatabase::Timestep::Timestep(double time, int nvars,
				     int totalParticles)
    : time(time), ids(totalParticles), data(nvars, totalParticles)
{
}

ParticleDatabase::IncompleteTimestep::IncompleteTimestep(int nproc)
    : ids(nproc), data(nproc), lock("Incomplete timestep lock")
{
    numReported=0;
}

ParticleDatabase::ParticleDatabase(PSECore::Dataflow::Module* module)
    : listenerlock("ParticleDatabase listener list lock"),
      varlock("ParticleDatabase variable lock"),
      waitStart("ParticleDatabase startup condition"),
      timestep_lock("ParticleDatbase timestep rw lock"),
      module(module)
{
}

ParticleDatabase::~ParticleDatabase()
{
}

// Simulation interface...
void ParticleDatabase::setup(const CIA::array1<CIA::string>& names)
{
    varlock.lock();
    varnames=names;
    waitStart.conditionSignal();
    varlock.unlock();
}

void ParticleDatabase::notifyGUI(double time)
{
    cerr << "timestep start: " << time << '\n';
    module->update_progress(0);
}

void ParticleDatabase::timestep(double time, int myrank, int nproc,
				const CIA::array1<int>& particleID,
				const CIA::array2<double>& data)
{
    varlock.lock();
    while(varnames.size() == 0)
	waitStart.wait(varlock);

    map<double, IncompleteTimestep*>::iterator iter=incomplete_timesteps.find(time);
    IncompleteTimestep* its;
    if(iter == incomplete_timesteps.end())
	incomplete_timesteps[time]=its=new IncompleteTimestep(nproc);
    else
	its=iter->second;
    varlock.unlock();

    its->ids[myrank]=particleID;
    its->data[myrank]=data;

    its->lock.lock();
    its->numReported++;
    module->update_progress(its->numReported, nproc);
    if(its->numReported == nproc){
	// We are the last one, make this a complete timestep and notify the world.

	// This sucks - consider trying to be smarter about already sorted data
	int totalParticles=0;
	for(int i=0;i<nproc;i++)
	    totalParticles+=(int)its->ids[i].size();
	cerr << "totalParticles: " << totalParticles << '\n';

	int nvars=(int)varnames.size();
#ifdef SUCKS
	cerr << "Sorting and processing...\n";
	int part=0;
	vector<int> ids(totalParticles);
	array2<double> data(varnames.size(), totalParticles);
	for(int p=0;p<nproc;p++){
	    vector<int>& pids=its->ids[p];
	    array2<double>& pdata=its->data[p];
	    int sp=part;
	    for(int i=0;i<(int)pids.size();i++){
		ids[part]=pids[i];
		part++;
	    }
	    for(int j=0;j<nvars;j++){
		part=sp;
		for(int i=0;i<(int)pids.size();i++){
		    data[j][part]=pdata[j][i];
		    part++;
		}
	    }
	}
	vector<int> map(totalParticles);
	generate(map.begin(), map.end(), linear(0));
	sort(map.begin(), map.end(), map_compare(ids));
	Timestep* ts=new Timestep(time, varnames.size(), totalParticles);
	for(int i=0;i<totalParticles;i++){
	    int s=map[i];
	    ts->ids[i]=ids[s];
	    for(int j=0;j<nvars;j++)
		ts->data[j][i]=data[j][s];
	}
#else
	Timestep* ts=new Timestep(time, varnames.size(), totalParticles);
	int part=0;
	for(int p=0;p<nproc;p++){
	    vector<int>& pids=its->ids[p];
	    array2<double>& pdata=its->data[p];
	    int sp=part;
	    for(int i=0;i<(int)pids.size();i++){
		ts->ids[part]=pids[i];
		part++;
	    }
	    for(int j=0;j<nvars;j++){
		part=sp;
		for(int i=0;i<(int)pids.size();i++){
		    ts->data[j][part]=pdata[j][i];
		    part++;
		}
	    }
	}
#endif

	timestep_lock.writeLock();
	timesteps.push_back(ts);
	timestep_lock.writeUnlock();

	cerr << "Notifying\n";
	listenerlock.lock();
	for(std::map<int, TimestepNotify>::iterator iter=listeners.begin();
	    iter != listeners.end(); iter++){
	    // Hmm..  This could cause problems if they try to
	    // manipulate the listener list during this callback...
	    iter->second->notifyNewTimestep(time);
	}
	listenerlock.unlock();

	// Destroy the incomplete timestep
	its->lock.unlock();
	delete its;

	// Remove the incomplete timestep from the database
	varlock.lock();
	incomplete_timesteps.erase(time);
	varlock.unlock();
	cerr << "All done\n";
    } else {
	its->lock.unlock();
    }
}


// Visualization interface...
CIA::array1<CIA::string> ParticleDatabase::listVariables()
{
    while(varnames.size() == 0){
	varlock.lock();
	waitStart.wait(varlock);
	varlock.unlock();
    }
    return varnames;
}

CIA::array1<double> ParticleDatabase::listTimesteps()
{
    CIA::array1<double> result;
    for(vector<Timestep*>::iterator iter=timesteps.begin();
	iter != timesteps.end();iter++)
	result.push_back((*iter)->time);
    return result;
}

void ParticleDatabase::getTimestep(double time, int dataIndex,
				   CIA::array1<int>& ids,
				   CIA::array2<double>& data)
{
    timestep_lock.readLock();
    vector<Timestep*>::iterator iter=timesteps.begin();
    for(;iter != timesteps.end();iter++){
	if((*iter)->time == time)
	    break;
    }
    timestep_lock.readUnlock();
    if(iter == timesteps.end()){
	cerr << "getTimestep, time " << time << " not found!\n";
    } else {
	ids=(*iter)->ids;
	data=(*iter)->data;
    }
}

void ParticleDatabase::getParticle(int index, CIA::array2<double>& data,
				   CIA::array1<double>& timesteps)
{
    NOT_FINISHED("void getParticle(int index, CIA::array2<double>& data");
}

int ParticleDatabase::registerNotify(const TimestepNotify& callback)
{
    /* REFERENCED */
    Guard autolock(&listenerlock);
    listeners[++nextID]=callback;
    return nextID;
}

void ParticleDatabase::unregisterNotify(int cbid)
{
    /* REFERENCED */
    Guard autolock(&listenerlock);
    map<int, TimestepNotify>::iterator iter=listeners.find(cbid);
    if(iter == listeners.end()){
	cerr << "unregister of " << cbid << "ignored - not found...\n";
	return;
    }
    iter->second=0; // Make sure it gets dereferenced...
    listeners.erase(cbid);
}

//
// $Log$
// Revision 1.5  2000/12/10 09:06:23  sparker
// Merge from csafe_risky1
//
// Revision 1.4  2000/12/06 21:49:47  kuzimmer
// Removing no longer needed legacy code
//
// Revision 1.3  2000/12/05 16:03:37  jas
// Remove g++ warnings.
//
// Revision 1.2  1999/10/15 20:23:00  sparker
// Mostly working
//
// Revision 1.1  1999/10/07 02:08:27  sparker
// use standard iostreams and complex type
//
//
