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
 *  TestStruct.cc:  Make an ImageField that fits the source field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/Fusion/Core/Datatypes/StructHexVolField.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class TestStruct : public Module
{
public:
  TestStruct(GuiContext *context);
  virtual ~TestStruct();

  virtual void execute();

private:
};


DECLARE_MAKER(TestStruct)


TestStruct::TestStruct(GuiContext *context)
  : Module("TestStruct", context, Filter, "Fields", "Fusion")
{
}



TestStruct::~TestStruct()
{
}


#define SIZEX 3
#define SIZEY 17
#define SIZEZ 5


void
TestStruct::execute()
{
  StructHexVolMesh *mesh = scinew StructHexVolMesh(SIZEX, SIZEY, SIZEZ);
  for (unsigned int k=0; k< SIZEZ; k++)
  {
    for (unsigned int j=0; j < SIZEY; j++)
    {
      for (unsigned int i = 0; i < SIZEX; i++)
      {
	StructHexVolMesh::Node::index_type index(i, j, k);

	Point point(i, sin(j*M_PI*2.0/(SIZEY-1))*(k+5),
		    cos(j*M_PI*2.0/(SIZEY-1))*(k+5));
	mesh->set_point(index, point);
      }
    }
  }

  StructHexVolField<double> *field =
    scinew StructHexVolField<double>(mesh, Field::NODE);
  

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Sample Field");
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  ofp->send(field);
}


} // End namespace SCIRun

