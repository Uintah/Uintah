
/*
 *  MPVisParticleSet.h:
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Datatypes_MPVizParticleSet_h
#define SCI_Datatypes_MPVizParticleSet_h 1

#include <Datatypes/Particles/ParticleSet.h>
#include <Datatypes/Particles/cfdlibParticleSet.h>
#include <Containers/LockingHandle.h>

namespace Uintah {
namespace Datatypes {



class MPVizParticleSet : public cfdlibParticleSet {
public:
    MPVizParticleSet(clString scalarVar, clString vectorVar,
		          void* cbClass);
    MPVizParticleSet(const MPVizParticleSet&);
    MPVizParticleSet();
    virtual ~MPVizParticleSet();

    virtual ParticleSet* clone() const;

    virtual void SetScalarId(const  clString& id);
    virtual void SetVectorId(const  clString& id);
  //  this is a hack.  cbClass must have a function called callback(int)
    virtual void SetCallback( void* cbClass);

    const clString& getScalarId();
    const clString& getVectorId();
  
    void *getCallbackClass();


    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;

  // testing
private:
  clString sVar;
  clString vVar;
  void* cbClass;
};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/07/27 16:58:59  mcq
// Initial commit
//
// Revision 1.1  1999/06/09 23:21:33  kuzimmer
// reformed the material/particle classes and removed the particleSetExtensions.  Now MPVizParticleSet inherits from cfdlibParticleSet-->use the new stl routines to dynamically check the particleSet type
//
// Revizion 1.1.1.1  1999/04/24 23:12:28  dav
// Import sources
//
//

#endif

