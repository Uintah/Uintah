
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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Geometry/Vector.h>

namespace Uintah {
namespace Datatypes {

using namespace SCICore::Datatypes;

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array1;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Geometry::Vector;

class ParticleSet;
typedef LockingHandle<ParticleSet> ParticleSetHandle;

class ParticleSet : public Datatype {
public:
    ParticleSet();
    virtual ~ParticleSet();
    ParticleSet(const ParticleSet&);

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

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.4  1999/09/21 16:08:31  kuzimmer
// modifications for binary file format
//
// Revision 1.3  1999/08/25 03:49:02  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:40:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:59:00  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 23:18:40  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif
