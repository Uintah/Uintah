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
 *  FieldCage.cc:
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>
#include <Core/Containers/Handle.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomDL.h>

#include <iostream>

namespace SCIRun {

class FieldCage : public Module {
public:
  FieldCage(GuiContext* ctx);
  virtual ~FieldCage();
  virtual void execute();

private:
  //! input port
  FieldIPort*              infield_;
  //! output port
  GeometryOPort           *ogeom_;  

  GuiInt		  sizex_;
  GuiInt		  sizey_;
  GuiInt		  sizez_;
  

  int                      field_generation_;
  int                      mesh_generation_;
  int			   fieldcage_id_;
  Vector                  *bounding_vector_;
  Point			  *bounding_min_;
};

DECLARE_MAKER(FieldCage)
FieldCage::FieldCage(GuiContext* ctx) : 
  Module("FieldCage", ctx, Filter, "Fields", "SCIRun"), 
  sizex_(ctx->subVar("sizex")),
  sizey_(ctx->subVar("sizey")),
  sizez_(ctx->subVar("sizez")),
  field_generation_(-1), 
  mesh_generation_(-1), 
  fieldcage_id_(0),
  bounding_vector_(0),
  bounding_min_(0),
  ogeom_(0),
  infield_(0)

{
}

FieldCage::~FieldCage()
{
}


void 
FieldCage::execute()
{
  // tell module downstream to delete everything we have sent it before.
  // This is typically viewer, it owns the scene graph memory we create here.
  
  
  
  infield_ = (FieldIPort *)get_iport("Field");
  if (!infield_) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  
  ogeom_ = (GeometryOPort *)get_oport("Scene Graph");
  if (!ogeom_) {
    error("Unable to initialize oport 'Scene Graph'.");
    return;
  }
  
  FieldHandle fld_handle;
  infield_->get(fld_handle);
  if(!fld_handle.get_rep())
  {
    warning("No Data in port 1 field.");
    return;
  }
  
  bool mesh_new = fld_handle->mesh()->generation != mesh_generation_;
  bool field_new = fld_handle->generation != field_generation_;
  if (field_new || mesh_new)
  {
    field_generation_  = fld_handle->generation;  
    mesh_generation_ = fld_handle->mesh()->generation; 
    if (bounding_vector_) delete bounding_vector_;
    if (bounding_min_) delete bounding_min_;
    bounding_vector_ = scinew Vector();
    bounding_min_ = scinew Point();
    *bounding_vector_ = fld_handle->mesh()->get_bounding_box().diagonal();
    *bounding_min_ = fld_handle->mesh()->get_bounding_box().min();
  }
  
  GeomCLines* lines = scinew GeomCLines;
  lines->setLineWidth(1.0);
  GeomSwitch *cage_switch = scinew GeomSwitch(scinew GeomDL(lines));
    
  const int xn = (sizex_.get() > 0 ? sizex_.get() : 2);
  const int yn = (sizey_.get() > 0 ? sizey_.get() : 2);
  const int zn = (sizez_.get() > 0 ? sizez_.get() : 2);
  int xi, yi, zi;
  const double dx = bounding_vector_->x() / (xn-1);
  const double dy = bounding_vector_->y() / (yn-1);
  const double dz = bounding_vector_->z() / (zn-1);
  const Point &min = *bounding_min_;
  MaterialHandle red = scinew Material(Color(1.0, 0.0, 0.0));
  MaterialHandle green = scinew Material(Color(0.0, 1.0, 0.0));
  MaterialHandle blue = scinew Material(Color(0.0, 0.0, 1.0));
  
  yi=0;
  for (xi = 0; xi < xn; xi++) {
    Point p1(min.x() + dx*xi, min.y() + dy*yi, min.z());
    Point p2(min.x()+dx*xi, min.y()+dy*yi, min.z()+bounding_vector_->z());
    lines->add(p1,blue,p2,blue);
  }
  yi=yn-1;
  for (xi = 0; xi < xn; xi++) {
    Point p1(min.x() + dx*xi, min.y() + dy*yi, min.z());
    Point p2(min.x()+dx*xi, min.y()+dy*yi, min.z()+bounding_vector_->z());
    lines->add(p1,blue,p2,blue);
  }
  xi=0;
  for (yi = 0; yi < yn; yi++) {
    Point p1(min.x() + dx*xi, min.y() + dy*yi, min.z());
    Point p2(min.x()+dx*xi, min.y()+dy*yi, min.z()+bounding_vector_->z());
    lines->add(p1,blue,p2,blue);
  }
  xi=xn-1;
  for (yi = 0; yi < yn; yi++) {
    Point p1(min.x() + dx*xi, min.y() + dy*yi, min.z());
    Point p2(min.x()+dx*xi, min.y()+dy*yi, min.z()+bounding_vector_->z());
    lines->add(p1,blue,p2,blue);
  }

  zi=0;
  for (xi = 0; xi < xn; xi++) {
    Point p1(min.x() + dx*xi, min.y(), min.z() + dz*zi);
    Point p2(min.x()+dx*xi, min.y()+bounding_vector_->y(), min.z()+dz*zi);
    lines->add(p1,green,p2,green);
  }
  zi=zn-1;
  for (xi = 0; xi < xn; xi++) {
    Point p1(min.x() + dx*xi, min.y(), min.z() + dz*zi);
    Point p2(min.x()+dx*xi, min.y()+bounding_vector_->y(), min.z()+dz*zi);
    lines->add(p1,green,p2,green);
  }
  xi=0;
  for (zi = 0; zi < zn; zi++) {
    Point p1(min.x() + dx*xi, min.y(), min.z() + dz*zi);
    Point p2(min.x()+dx*xi, min.y()+bounding_vector_->y(), min.z()+dz*zi);
    lines->add(p1,green,p2,green);
  }
  xi=xn-1;
  for (zi = 0; zi < zn; zi++) {
    Point p1(min.x() + dx*xi, min.y(), min.z() + dz*zi);
    Point p2(min.x()+dx*xi, min.y()+bounding_vector_->y(), min.z()+dz*zi);
    lines->add(p1,green,p2,green);
  }

  zi=0;
  for (yi = 0; yi < yn; yi++) {
    Point p1(min.x(), min.y() + dy*yi, min.z() + dz*zi);
    Point p2(min.x()+bounding_vector_->x(), min.y()+dy*yi, min.z()+dz*zi);
    lines->add(p1,red,p2,red);
  }
  zi=zn-1;
  for (yi = 0; yi < yn; yi++) {
    Point p1(min.x(), min.y() + dy*yi, min.z() + dz*zi);
    Point p2(min.x()+bounding_vector_->x(), min.y()+dy*yi, min.z()+dz*zi);
    lines->add(p1,red,p2,red);
  }
  yi=0;
  for (zi = 0; zi < zn; zi++) {
    Point p1(min.x(), min.y() + dy*yi, min.z() + dz*zi);
    Point p2(min.x()+bounding_vector_->x(), min.y()+dy*yi, min.z()+dz*zi);
    lines->add(p1,red,p2,red);
  }
  yi=yn-1;
  for (zi = 0; zi < zn; zi++) {
    Point p1(min.x(), min.y() + dy*yi, min.z() + dz*zi);
    Point p2(min.x()+bounding_vector_->x(), min.y()+dy*yi, min.z()+dz*zi);
    lines->add(p1,red,p2,red);
  }

  const char *name = "Field Cage";
  if (fieldcage_id_) ogeom_->delObj(fieldcage_id_);
  fieldcage_id_ = ogeom_->addObj(cage_switch, name);

  ogeom_->flushViews();
}



} // End namespace SCIRun


