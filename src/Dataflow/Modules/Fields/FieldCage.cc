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
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
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

  int                     old_x_, old_y_, old_z_;

  int                      field_generation_;
  int                      mesh_generation_;
  int			   fieldcage_id_;
  Vector                  *bounding_vector_;
  Point			  *bounding_min_;

  void add_lines(const Point &p1, const Point &p2, const Point &p3, 
		 const  MaterialHandle &mat, GeomLines* lines, 
		 const double incr, const int boxes);

};


DECLARE_MAKER(FieldCage)
FieldCage::FieldCage(GuiContext* ctx) : 
  Module("FieldCage", ctx, Filter, "FieldsOther", "SCIRun"), 
  infield_(0),
  ogeom_(0),
  sizex_(ctx->subVar("sizex")),
  sizey_(ctx->subVar("sizey")),
  sizez_(ctx->subVar("sizez")),
  old_x_(-1), old_y_(-1), old_z_(-1),
  field_generation_(-1), 
  mesh_generation_(-1), 
  fieldcage_id_(0),
  bounding_vector_(0),
  bounding_min_(0)
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
  ogeom_ = (GeometryOPort *)get_oport("Scene Graph");
  
  FieldHandle fld_handle;
  infield_->get(fld_handle);
  if(!fld_handle.get_rep())
  {
    warning("No Data in port 1 field.");
    return;
  }
  
  bool mesh_new = fld_handle->mesh()->generation != mesh_generation_;
  bool field_new = fld_handle->generation != field_generation_;
  vector<unsigned int> min, max;
  Point p1, p2, p3, p4, p5, p6, p7, p8;

  const int xn = (sizex_.get() > 0 ? sizex_.get() : 2);
  const int yn = (sizey_.get() > 0 ? sizey_.get() : 2);
  const int zn = (sizez_.get() > 0 ? sizez_.get() : 2);

  const double x_incr = 1.0/(xn-1);
  const double y_incr = 1.0/(yn-1);
  const double z_incr = 1.0/(zn-1);
  

  if (field_new || mesh_new || old_x_ != sizex_.get() || old_y_ != sizey_.get() ||
      old_z_ != sizez_.get())
  {
    field_generation_  = fld_handle->generation;  
    mesh_generation_ = fld_handle->mesh()->generation; 
    old_x_ = sizex_.get();
    old_y_ = sizey_.get();
    old_z_ = sizez_.get();
    if (bounding_vector_) delete bounding_vector_;
    if (bounding_min_) delete bounding_min_;

    GeomLines* lines = scinew GeomLines;
    lines->setLineWidth(1.0);
    GeomSwitch *cage_switch = scinew GeomSwitch(scinew GeomDL(lines));

    MaterialHandle red = scinew Material(Color(1.0, 0.0, 0.0));
    MaterialHandle green = scinew Material(Color(0.0, 1.0, 0.0));
    MaterialHandle blue = scinew Material(Color(0.0, 0.0, 1.0));

    // for Image and LatVols have the field cage coordinates 
    // correspond to the exact corner cooords
    string field_type = fld_handle->get_type_description(0)->get_name();
    if( field_type.find("LatVolField") != string::npos) {
      // use min coords of latvol
      LatVolMesh *m = (LatVolMesh*) fld_handle->mesh().get_rep();
      LatVolMesh::Node::index_type index;

      m->get_min(min);
      m->get_dim(max);
      max[0] = max[0]-1;
      max[1] = max[1]-1;
      max[2] = max[2]-1;

      index.i_ = min[0];
      index.j_ = min[1];
      index.k_ = min[2];
      m->get_point(p1, index);

      index.i_ = max[0];
      index.j_ = min[1];
      index.k_ = min[2];
      m->get_point(p2,index);

      index.i_ = max[0];
      index.j_ = max[1];
      index.k_ = min[2];
      m->get_point(p3,index);

      index.i_ = min[0];
      index.j_ = max[1];
      index.k_ = min[2];
      m->get_point(p4,index);

      index.i_ = min[0];
      index.j_ = min[1];
      index.k_ = max[2];
      m->get_point(p5,index);

      index.i_ = max[0];
      index.j_ = min[1];
      index.k_ = max[2];
      m->get_point(p6,index);

      index.i_ = max[0];
      index.j_ = max[1];
      index.k_ = max[2];
      m->get_point(p7,index);

      index.i_ = min[0];
      index.j_ = max[1];
      index.k_ = max[2];
      m->get_point(p8,index);

      add_lines(p1, p2, p4, red, lines, x_incr, xn-1);
      add_lines(p5, p6, p8, red, lines, x_incr, xn-1);
      add_lines(p4, p3, p8, red, lines, x_incr, xn-1);
      add_lines(p1, p2, p5, red, lines, x_incr, xn-1);

      add_lines(p2, p3, p1, green, lines, y_incr, yn-1);
      add_lines(p6, p7, p2, green, lines, y_incr, yn-1);
      add_lines(p5, p8, p1, green, lines, y_incr, yn-1);
      add_lines(p6, p7, p5, green, lines, y_incr, yn-1);

      add_lines(p2, p6, p3, blue, lines, z_incr, zn-1);
      add_lines(p3, p7, p4, blue, lines, z_incr, zn-1);
      add_lines(p2, p6, p1, blue, lines, z_incr, zn-1);
      add_lines(p1, p5, p4, blue, lines, z_incr, zn-1);
    } else if( field_type.find("ImageField") != string::npos) {
      // use min coords of image
      ImageMesh *m = (ImageMesh*) fld_handle->mesh().get_rep();
      ImageMesh::Node::index_type index;

      m->get_min(min);
      min.push_back(0);
      m->get_dim(max);
      max.push_back(0);
      max[0] = max[0]-1;
      max[1] = max[1]-1;

      index.i_ = min[0];
      index.j_ = min[1];
      m->get_point(p1, index);

      index.i_ = max[0];
      index.j_ = min[1];
      m->get_point(p2,index);

      index.i_ = max[0];
      index.j_ = max[1];
      m->get_point(p3,index);

      index.i_ = min[0];
      index.j_ = max[1];
      m->get_point(p4,index);

      // FIX ME: determine if a plane in x, y, or z
      add_lines(p1, p2, p4, red, lines, x_incr, xn-1);
      add_lines(p2, p3, p1, green, lines, y_incr, yn-1);
    } else {
      bounding_vector_ = scinew Vector();
      bounding_min_ = scinew Point();

      *bounding_vector_ = fld_handle->mesh()->get_bounding_box().diagonal();
      *bounding_min_ = fld_handle->mesh()->get_bounding_box().min();
   
      
      int xi, yi, zi;
      const double dx = bounding_vector_->x() / (xn-1);
      const double dy = bounding_vector_->y() / (yn-1);
      const double dz = bounding_vector_->z() / (zn-1);
      const Point &min = *bounding_min_;
      
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
    const char *name = "Field Cage";
    if (fieldcage_id_) ogeom_->delObj(fieldcage_id_);
    fieldcage_id_ = ogeom_->addObj(cage_switch, name);
    
    ogeom_->flushViews();
  } 

} 

void
FieldCage::add_lines(const Point &p1, const Point &p2, const Point &p3, 
		     const  MaterialHandle &mat, GeomLines* lines, 
		     const double incr, const int boxes) {

  Vector y1(p3.x()-p1.x(), p3.y()-p1.y(), p3.z()-p1.z());
  
  Point t1(p1.x(), p1.y(), p1.z());
  Point t2(p2.x(), p2.y(), p2.z());

  lines->add(t1,mat,t2,mat);
  
  for(int count=0; count<boxes; count++) {
    t1.x(t1.x()+incr*y1[0]);
    t1.y(t1.y()+incr*y1[1]);
    t1.z(t1.z()+incr*y1[2]);

    t2.x(t2.x()+incr*y1[0]);
    t2.y(t2.y()+incr*y1[1]);
    t2.z(t2.z()+incr*y1[2]);
    lines->add(t1,mat,t2,mat);
  }

}


} // End namespace SCIRun


