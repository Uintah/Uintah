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
 *  MoveElemToNode.cc:  Rotate and flip field to get it into "standard" view
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
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>

namespace SCIRun {


class MoveElemToNode : public Module
{
public:
  MoveElemToNode(GuiContext* ctx);
  virtual ~MoveElemToNode();

  virtual void execute();

protected:
  int ifield_generation_;
};


DECLARE_MAKER(MoveElemToNode)

MoveElemToNode::MoveElemToNode(GuiContext* ctx)
  : Module("MoveElemToNode", ctx, Filter, "Fields", "SCIRun"),
    ifield_generation_(0)
{
}


MoveElemToNode::~MoveElemToNode()
{
}


void
MoveElemToNode::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Elem Field");
  FieldHandle ifield;
  if (!ifp) {
    error("Unable to initialize iport 'Elem Field'.");
    return;
  }
  if (!(ifp->get(ifield) && ifield.get_rep()))
  {
    return;
  }

  if (ifield_generation_ != ifield->generation)
  {
    ifield_generation_ = ifield->generation;

    LatVolMesh *imesh = dynamic_cast<LatVolMesh *>(ifield->mesh().get_rep());

    const int ni = imesh->get_ni();
    const int nj = imesh->get_nj();
    const int nk = imesh->get_nk();

    const double ioff = (1.0 - ((ni-2.0) / (ni-1.0))) * 0.5;
    const double joff = (1.0 - ((nj-2.0) / (nj-1.0))) * 0.5;
    const double koff = (1.0 - ((nk-2.0) / (nk-1.0))) * 0.5;
    cout << "offsets: " << ioff << " " << joff << " " << koff << "\n";
    const Point minp(ioff, joff, koff);
    const Point maxp(1.0-ioff, 1.0-joff, 1.0-koff);

    LatVolMesh *omesh = scinew LatVolMesh(ni-1, nj-1, nk-1, minp, maxp);

    Transform trans;
    imesh->get_canonical_transform(trans);
    omesh->transform(trans);

    LatVolField<double> *ofield =
      scinew LatVolField<double>(omesh, Field::CELL);

    FieldOPort *ofp = (FieldOPort *)get_oport("Node Field");
    if (!ofp) {
      error("Unable to initialize oport 'Node Field'.");
      return;
    }

    ofp->send(FieldHandle(ofield));
  }
}


} // End namespace SCIRun

