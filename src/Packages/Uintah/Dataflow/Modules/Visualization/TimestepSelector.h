/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


/****************************************
CLASS
    TimestepSelector


OVERVIEW TEXT
    This module receives a DataArchive and selects the visualized timestep.
    Or Animates the data.



KEYWORDS
    

AUTHOR
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June 2000

    Copyright (C) 1999 SCI Group

LOG
    Created June 26, 2000
****************************************/
#ifndef TIMESTEPSELECTOR_H
#define TIMESTEPSELECTOR_H 1


#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Datatypes/ArchivePort.h>
#include <Dataflow/Network/Module.h> 
#include <Core/GuiInterface/GuiVar.h> 
#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>


namespace Uintah {

using namespace SCIRun;

class TimestepSelector : public Module { 
  
public: 

  // GROUP: Constructors
  //////////
  TimestepSelector(const string& id); 

  // GROUP: Destructors
  //////////
  virtual ~TimestepSelector(); 

  // GROUP: cloning and execution 
  ////////// 
  virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

protected:
  
private:

  GuiString tcl_status;

  GuiInt animate;
  GuiInt anisleep;
  GuiInt time;
  GuiDouble timeval;

  ArchiveIPort *in;
  ArchiveOPort *out;
  
  ArchiveHandle archiveH;
  void setVars(ArchiveHandle ar);

}; //class 
} // End namespace Uintah

#endif
