#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/Persistent.h>

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>

namespace Uintah {

using namespace SCIRun;

/**************************************

CLASS
   Archive
   
   Simple Archive Class.

GENERAL INFORMATION

   Archive.h

   Packages/Kurt Zimmerman
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
} // End namespace Uintah

#endif
