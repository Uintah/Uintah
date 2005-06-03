/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

#ifndef SLAVEDATABASE_H
#define SLAVEDATABASE_H

//
// SlaveDatabase.h
//
// This class is used by the MasterController to manage information 
// about which SlaveControllers have been created and what there
// status is.
//

#include <testprograms/Component/dav/Common/MastContInterfaces_sidl.h>

#include <Core/Exceptions/InternalError.h>

#include <Core/CCA/SSIDL/array.h>

#include <string>

namespace dav {

using namespace SCIRun;
using std::string;
using SSIDL::array1;

class SlaveInfo {
  
public:
  SlaveInfo( string machineId, Mc2Sc mc2Sc, int maxProcs );

  string       d_machineId;
  Mc2Sc        d_mc2Sc;
  int          d_maxProcs;  // Number of processors on this SC's machine.
};

class SlaveDatabase {
public:

  SlaveDatabase();
  ~SlaveDatabase();

  int         numActive();
  SlaveInfo * find( string machineId );
  SlaveInfo * leastLoaded();
  void        add( SlaveInfo * si ) throw (InternalError);
  void        remove( SlaveInfo * si );

  void        getMachineIds( array1<string> & ids );

  void        updateGuis();  // Runs through list of SCs and tells
                             // them to send gui info to the gui

  void        shutDown();  // Runs through the list of each of the active
                           // SCs and sends a shutDown message to them.

private:

  array1<SlaveInfo*>::iterator   findPosition( string machineId );

  int                 d_numSlaves;
  array1<SlaveInfo *> d_slaves;
};

} // end namespace dav

#endif
