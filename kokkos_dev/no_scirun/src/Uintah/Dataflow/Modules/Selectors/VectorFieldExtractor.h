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
    VectorFieldExtractor

    

OVERVIEW TEXT
    This module receives a DataArchive object.  The user
    interface is dynamically created based information provided by the
    DataArchive.  The user can then select which variables he/she
    wishes to view in a visualization.



KEYWORDS
    ParticleGridReader, Material/Particle Method

AUTHOR
    Packages/Kurt Zimmerman
    Department of Computer Science
    University of Utah
    June, 2000

    Copyright (C) 2000 SCI Group

LOG
    Created June 27, 2000
****************************************/
#ifndef VECTORFIELDEXTRACTOR_H
#define VECTORFIELDEXTRACTOR_H 1


#include <Uintah/Core/Datatypes/Archive.h>
#include <Uintah/Dataflow/Ports/ArchivePort.h>
#include <Uintah/Dataflow/Modules/Selectors/FieldExtractor.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h> 
#include <string>
#include <vector>


namespace Uintah {
using namespace SCIRun;

class VectorFieldExtractor : public FieldExtractor { 
  
public: 

  // GROUP: Constructors
  //////////
  VectorFieldExtractor(GuiContext* ctx);

  // GROUP: Destructors
  //////////
  virtual ~VectorFieldExtractor(); 

  // GROUP: cloning and execution 
  ////////// 
//    virtual void execute(); 

  //////////
  // overides tcl_command in base class Module
  //  void tcl_command( TCLArgs&, void* );

protected:
  virtual void get_vars(vector< string >&,
			vector< const TypeDescription *>&);
 
private:


}; //class 
} // End namespace Uintah



#endif
