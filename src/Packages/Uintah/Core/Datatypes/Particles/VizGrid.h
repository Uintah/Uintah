#ifndef SCI_Datatypes_VizGrid_h
#define SCI_Datatypes_VizGrid_h 1



/*----------------------------------------------------------------------
CLASS
    VizGrid

    A container class for data.

OVERVIEW TEXT
    VizGrid contains scalar fields, vector fields and particle sets that
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

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Uintah/Datatypes/Particles/ParticleSet.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/VectorField.h>
#include <map.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/String.h>

namespace Uintah {
namespace Datatypes {

using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
class  VizGrid;
typedef LockingHandle<VizGrid> VizGridHandle;
  
class VizGrid : public Datatype {
public:
  VizGrid();
  virtual ~VizGrid();

  virtual VizGrid* clone() const=0;

  virtual void AddVectorField(const clString& name, VectorFieldHandle vfh) =0;
  virtual void AddScalarField(const clString& name, ScalarFieldHandle sfh)=0;

  virtual VectorFieldHandle getVectorField( clString name ) =0;
  virtual ScalarFieldHandle getScalarField( clString name ) =0;
  
  virtual void getScalarNames( Array1< clString>&)=0;
  virtual void getVectorNames( Array1< clString>&)=0;
  virtual clString getName()=0;

  ////////// Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;


private:


};

} // End namespace Datatypes
} // End namespace Uintah

//
// $Log$
// Revision 1.1  1999/09/21 16:08:31  kuzimmer
// modifications for binary file format
//



#endif
