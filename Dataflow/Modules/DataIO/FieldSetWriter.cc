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
 *  FieldSetWriter.cc: Save persistent representation of a field to a file
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Ports/FieldSetPort.h>
#include <Dataflow/Modules/DataIO/GenericWriter.h>

namespace SCIRun {

template class GenericWriter<FieldSetHandle>;

class FieldSetWriter : public GenericWriter<FieldSetHandle> {
public:
  FieldSetWriter(const string& id);
};


extern "C" Module* make_FieldSetWriter(const string& id) {
  return new FieldSetWriter(id);
}


FieldSetWriter::FieldSetWriter(const string& id)
  : GenericWriter<FieldSetHandle>("FieldSetWriter", id, "DataIO", "SCIRun")
{
}


} // End namespace SCIRun
