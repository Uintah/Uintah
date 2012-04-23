/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef ARCHIVE_H
#define ARCHIVE_H

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/Persistent.h>

#include <Core/DataArchive/DataArchive.h>

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

  // This is defined here, so that we don't have to include
  // LockingHandle and Persistent stuff in DataArchive.
  typedef LockingHandle<DataArchive> DataArchiveHandle;
  
class Archive : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  Archive(const DataArchiveHandle& archive);
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
  DataArchiveHandle operator()(){ return archive; };
  DataArchiveHandle getDataArchive() { return archive; };
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
  DataArchiveHandle archive;
  int _timestep;
};
} // End namespace Uintah

#endif
