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
 *  Clip.cc:  Rotate and flip field to get it into "standard" view
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
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/BoxClipper.h>
#include <iostream>

namespace SCIRun {


class Clip : public Module
{
private:
  ScaledBoxWidget *box_;
  CrowdMonitor widget_lock_;
  BBox last_bounds_;
  GuiInt mode_;
  int  last_generation_;

public:
  Clip(const string& id);
  virtual ~Clip();

  virtual void execute();

};


extern "C" Module* make_Clip(const string& id) {
  return new Clip(id);
}


Clip::Clip(const string& id)
  : Module("Clip", id, Source, "Fields", "SCIRun"),
    widget_lock_("Clip widget lock"),
    mode_("runmode", id, this),
    last_generation_(0)
{
  box_ = scinew ScaledBoxWidget(this, &widget_lock_, 1.0, 1);
}


Clip::~Clip()
{
}


void
Clip::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }

  // update the widget
  BBox obox = ifieldhandle->mesh()->get_bounding_box();
  if (!(last_bounds_.valid() && obox.valid() &&
	obox.min() == last_bounds_.min() &&
	obox.max() == last_bounds_.max()))
  {
    const BBox bbox = ifieldhandle->mesh()->get_bounding_box();
    const Point &bmin = bbox.min();
    const Point &bmax = bbox.max();

    const Point center = bmin + Vector(bmax - bmin) * 0.25;
    const Point right = center + Vector((bmax.x()-bmin.x())/4.0, 0, 0);
    const Point down = center + Vector(0, (bmax.y()-bmin.y())/4.0, 0);
    const Point in = center + Vector(0, 0, (bmax.z()-bmin.z())/4.0);

    const double l2norm = (bmax - bmin).length();

    box_->SetScale(l2norm * 0.015);
    box_->SetPosition(center, right, down, in);

    GeomGroup *widget_group = scinew GeomGroup;
    widget_group->add(box_->GetWidget());

    GeometryOPort *ogport=0;
    ogport = (GeometryOPort*)get_oport("Selection Widget");
    if (!ogport) {
      postMessage("Unable to initialize "+name+"'s oport\n");
      return;
    }
    ogport->addObj(widget_group, "Clip Selection Widget",
		   &widget_lock_);
    ogport->flushViews();

    last_bounds_ = obox;
  }
  
  if (mode_.get() == 1 || mode_.get() == 2)
  {
    BoxClipper clipper = box_->get_clipper();
    TetVolMeshHandle omesh = (TetVolMesh *)(ifieldhandle->mesh().get_rep());
    TetVolMeshHandle nmesh = (TetVolMesh *)(omesh->clip(clipper).get_rep());
    TetVol<double> *ofield =
      scinew TetVol<double>(nmesh, ifieldhandle->data_at());

    FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
    if (!ofield_port) {
      postMessage("Unable to initialize "+name+"'s oport\n");
      return;
    }
    
    ofield_port->send(ofield);
  }
}



} // End namespace SCIRun

