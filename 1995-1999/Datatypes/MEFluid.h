#ifndef SCI_Datatypes_MEFluid_h
#define SCI_Datatypes_MEFluid_h 1


/*----------------------------------------------------------------------
CLASS
    MEFluid

    A container class for data.

OVERVIEW TEXT
    MEFluid contains scalar fields, vector fields and particle sets that
    from Mechanical Engineering simulations.


KEYWORDS


AUTHOR
    Kurt Zimmerman
    Department of Computer Science
    University of Utah
    January 1999

    Copyright (C) 1999 SCI Group

LOG
    Created January 12, 1999
----------------------------------------------------------------------*/

#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Datatypes/ParticleSet.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/VectorField.h>
#include <Classlib/Array1.h>
#include <Classlib/String.h>


class MEFluid : public Datatype {
public:
  MEFluid();
  virtual ~MEFluid();
  MEFluid(const MEFluid&);
  virtual MEFluid* clone() const;

  void AddVectorField(const clString& name, VectorFieldHandle vfh);
  void AddScalarField(const clString& name, ScalarFieldHandle sfh);
  void AddParticleSet(ParticleSetHandle psh);

  ParticleSetHandle getParticleSet();
  VectorFieldHandle getVectorField( clString name );
  ScalarFieldHandle getScalarField( clString name );
  
  void getScalarVars( Array1< clString>&);
  void getVectorVars( Array1< clString>&);
  

  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;


private:
  Array1< clString > scalarvars;
  Array1< clString > vectorvars;
  Array1< VectorFieldHandle > vfs;
  Array1< ScalarFieldHandle > sfs;
  
  ParticleSetHandle ps;
};

#endif
