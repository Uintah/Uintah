
/*
 *  ParticleSet.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_Datatypes_ParticleSet_h
#define SCI_Datatypes_ParticleSet_h 1

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/Array1.h>
#include <Geometry/Vector.h>

class ParticleSet;
typedef LockingHandle<ParticleSet> ParticleSetHandle;

class ParticleSet : public Datatype {
public:
    ParticleSet();
    virtual ~ParticleSet();
    ParticleSet(const ParticleSet&);
    virtual ParticleSet* clone() const=0;

    virtual int find_scalar(const clString& name)=0;
    virtual void list_scalars(Array1<clString>& names)=0;
    virtual int find_vector(const clString& name)=0;
    virtual void list_vectors(Array1<clString>& names)=0;

    virtual void list_natural_times(Array1<double>& times)=0;

    virtual void get(int timestep, int vectorid, Array1<Vector>& value,
		     int start=-1, int end=-1)=0;
    virtual void get(int timestep, int scalarid, Array1<double>& value,
		     int start=-1, int end=-1)=0;

    virtual Vector getVector(int timestep, int vectorid, int index)=0;
    virtual double getScalar(int timestep, int scalarid, int index)=0;

    virtual void interpolate(double time, int vectorid, Vector& value,
			     int start=-1, int end=-1)=0;
    virtual void interpolate(double time, int scalarid, double& value,
			     int start=-1, int end=-1)=0;

    virtual int position_vector()=0;

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

#endif
