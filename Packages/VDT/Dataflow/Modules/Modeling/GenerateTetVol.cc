/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  GenerateTetVol.cc:
 *
 *  Written by:
 *   mcole
 *   TODAY'S DATE HERE
 *
 */

#include <sci_defs.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/VDT/share/share.h>

namespace VDT {
extern "C" {
#include <vdtpub.h>
}

using namespace SCIRun;

class VDTSHARE GenerateTetVol : public Module {
public:
  GenerateTetVol(const string& id);

  virtual ~GenerateTetVol();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" VDTSHARE Module* make_GenerateTetVol(const string& id) {
  return scinew GenerateTetVol(id);
}

GenerateTetVol::GenerateTetVol(const string& id)
  : Module("GenerateTetVol", id, Source, "Modeling", "VDT")
{
}

GenerateTetVol::~GenerateTetVol(){
}

void GenerateTetVol::execute()
{
  // just test that the lib is loaded and we can make a call into it for now.
  VDT gentv_vdt = VDT_new_mesher();
  cout << "completed" << endl;

}

void GenerateTetVol::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace VDT


