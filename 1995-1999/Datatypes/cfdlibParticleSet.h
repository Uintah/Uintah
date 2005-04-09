
/*
 *  cfdlibParticleSet.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1998 SCI Group
 */

#ifndef SCI_Datatypes_cfdlibParticleSet_h
#define SCI_Datatypes_cfdlibParticleSet_h 1

#include <Datatypes/ParticleSet.h>
#include <Classlib/LockingHandle.h>

// struct cfdlibTimeStep {
//     double time;
//     Array1<Vector> positions;
//     Array1<double> scalars;
// };

struct cfdlibTimeStep {
  double time;
  Array1< Array1< Vector > > vectors;
  Array1< Array1< double > > scalars;
};

class cfdlibParticleSet : public ParticleSet {
  Array1<cfdlibTimeStep*> timesteps;
  Array1<clString> vectorvars;
  Array1<clString> scalarvars;
public:
    cfdlibParticleSet();
    virtual ~cfdlibParticleSet();

    virtual ParticleSet* clone() const;

    virtual int find_scalar(const clString& name);
    virtual void list_scalars(Array1<clString>& names);
    virtual int find_vector(const clString& name);
    virtual void list_vectors(Array1<clString>& names);

    virtual void list_natural_times(Array1<double>& times);

    virtual void get(int timestep, int vectorid, Array1<Vector>& value,
		     int start=-1, int end=-1);
    virtual void get(int timestep, int scalarid, Array1<double>& value,
		     int start=-1, int end=-1);

    virtual Vector getVector(int timestep, int vectorid, int index);
    virtual double getScalar(int timestep, int scalarid, int index);

    virtual void interpolate(double time, int vectorid, Vector& value,
			     int start=-1, int end=-1);
    virtual void interpolate(double time, int scalarid, double& value,
				     int start=-1, int end=-1);
    virtual int position_vector();

    void add(cfdlibTimeStep* ts);
    void addVectorVar(const clString&);
    void addScalarVar(const clString&);
    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;

  // testing
  void print();
};

#endif
