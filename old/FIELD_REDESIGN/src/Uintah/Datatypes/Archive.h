#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Uintah/Interface/DataArchive.h>
#include <SCICore/Persistent/Persistent.h>
#include <iostream>

namespace Uintah {
namespace Datatypes {

using SCICore::Datatypes::Datatype;
using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;



/**************************************

CLASS
   Archive
   
   Simple Archive Class.

GENERAL INFORMATION

   Archive.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   Archive class.
  
WARNING
  
****************************************/
class Archive;
typedef LockingHandle<Archive> ArchiveHandle;

class Archive : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  Archive(DataArchive *archive);
  //////////
  // Constructor
  Archive();
  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~Archive();
 
  // GROUP: Access
  //////////
  // return the archive
  DataArchive* operator()(){ return archive; };
  //////////
  // return the selected timestep
  int timestep(){ return _timestep; }
  
  // GROUP: Modify
  //////////  
  // Set the timestep
  void SetTimestep( int t ){ _timestep = t; }

  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  

private:
  DataArchive *archive;
  int _timestep;
};

} // end namespace Datatypes
} // end namespace Uintah
#endif
