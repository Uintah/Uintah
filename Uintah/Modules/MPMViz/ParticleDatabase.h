
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

#ifndef Uintah_Modules_MPMViz_ParticleDatabase_h
#define Uintah_Modules_MPMViz_ParticleDatabase_h 1

#include <Uintah/Datatypes/Particles/Particles_sidl.h>
#include <SCICore/Thread/ConditionVariable.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <SCICore/Thread/Mutex.h>
#include <map>

namespace Uintah {
namespace Modules {

    using Uintah::Datatypes::Particles::Database_interface;
    using Uintah::Datatypes::Particles::TimestepNotify;

class ParticleDatabase : public Database_interface {
    std::map<int, TimestepNotify> listeners;
    int nextID;
    SCICore::Thread::Mutex listenerlock;

    SCICore::Thread::Mutex varlock;
    std::vector<CIA::string> varnames;

    SCICore::Thread::ConditionVariable waitStart;
    struct Timestep {
	Timestep(double time, int numVars, int totalParticles);
	double time;
	std::vector<int> ids;
	CIA::array2<double> data;
    };

    struct IncompleteTimestep {
	std::vector<std::vector<int> > ids;
	std::vector<CIA::array2<double> > data;
	SCICore::Thread::Mutex lock;
	IncompleteTimestep(int nproc);
	int numReported;
    };

    class linear {
	int i;
    public:
	linear(int i) : i(i) {}
	int operator()() {
	    return i++;
	}
    };

    class map_compare {
	std::vector<int>& ids;
    public:
	map_compare(std::vector<int>& ids) : ids(ids) {}
	int operator()(int i, int j) {
	    return ids[i] < ids[j];
	}
    };

    std::map<double, IncompleteTimestep*> incomplete_timesteps;

    SCICore::Thread::CrowdMonitor timestep_lock;
    std::vector<Timestep*> timesteps;

public:
    ParticleDatabase();
    virtual ~ParticleDatabase();

    // Simulation interface...
    virtual void setup(const CIA::array1<CIA::string>& names);
    virtual void notifyGUI(double time);
    virtual void timestep(double time, int myrank, int nproc,
			  const CIA::array1<int>& particleID,
			  const CIA::array2<double>& data);

    // Visualization interface...
    virtual CIA::array1<CIA::string> listVariables();
    virtual CIA::array1<double> listTimesteps();
    virtual void getTimestep(double time, int dataIndex,
			     CIA::array1<int>& ids,
			     CIA::array2<double>& data);
    virtual void getParticle(int index, CIA::array2<double>& data,
			     CIA::array1<double>& timesteps);
    virtual int registerNotify(const TimestepNotify& callback);
    virtual void unregisterNotify(int);
};

} // End namespace Modules
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/10/07 02:08:27  sparker
// use standard iostreams and complex type
//
//

#endif
