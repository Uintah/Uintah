#ifndef SCI_Datatypes_ParticleSetExtension_h
#define SCI_Datatypes_ParticleSetExtension_h 1


/*----------------------------------------------------------------------
CLASS
    ParticleSetExtension

    A class for passing extra data related to a particle set

OVERVIEW TEXT

KEYWORDS

AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 6, 1999
----------------------------------------------------------------------*/


#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/Array1.h>
#include <Datatypes/MEFluid.h>
#include <Geometry/Vector.h>

class ParticleSetExtension;
typedef LockingHandle<ParticleSetExtension> ParticleSetExtensionHandle;


class ParticleSetExtension : public Datatype {
public:
    ParticleSetExtension();
  //  this is a hack.  cbClass must have a function called callback(int)
    ParticleSetExtension( clString scalarVar, clString vectorVar,
		          void* cbClass);
    virtual ~ParticleSetExtension();
    ParticleSetExtension(const ParticleSetExtension&);
    virtual ParticleSetExtension* clone() const;


    virtual void SetScalarId(const  clString& id);
    virtual void SetVectorId(const  clString& id);
  //  this is a hack.  cbClass must have a function called callback(int)
    virtual void SetCallback( void* cbClass);

    const clString& getScalarId();
    const clString& getVectorId();
  
    void *getCallbackClass();

  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  clString sVar;
  clString vVar;
  void* cbClass;

};

#endif
