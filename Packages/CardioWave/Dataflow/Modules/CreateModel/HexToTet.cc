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
 *  HexToTet.cc:  Clip out the portions of a HexVol with specified values
 *
 *  Written by:
 *   Michael Callahan
 *   University of Utah
 *   May 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <vector>
#include <algorithm>


namespace CardioWave {

using std::cerr;
using std::endl;

using namespace SCIRun;

class HexToTet : public Module {
  
public:

  //! Constructor/Destructor
  HexToTet(GuiContext *context);
  virtual ~HexToTet();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(HexToTet)


HexToTet::HexToTet(GuiContext *context) : 
  Module("HexToTet", context, Filter, "CreateModel", "CardioWave")
{
}


HexToTet::~HexToTet()
{
}


void
HexToTet::execute()
{
  FieldIPort *ifieldport = (FieldIPort *)get_iport("HexVol");
  if (!ifieldport) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  FieldHandle ifieldhandle;
  if(!(ifieldport->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Can't get field.");
    return;
  }

  HexVolField<int> *hvfield =
    dynamic_cast<HexVolField<int> *>(ifieldhandle.get_rep());

  if (hvfield == 0)
  {
    error("'" + ifieldhandle->get_type_description()->get_name() + "'" +
	  " field type is unsupported.");
    return;
  }

  TetVolField<int> *tvfield = 0;

  // Forward the results.
  FieldOPort *ofp = (FieldOPort *)get_oport("Masked HexVol");
  if (!ofp)
  {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }
  ofp->send(tvfield);
}


} // End namespace CardioWave
