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
 *  BuildSeedVolume.cc:
 *
 *  Written by:
 *   darbyb
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/PointCloudField.h>

#include <Packages/Insight/share/share.h>

namespace Insight {
  
using namespace SCIRun;
  
class InsightSHARE BuildSeedVolume : public Module {  
public:
  GuiDouble  gui_inside_value_;
  GuiDouble  gui_outside_value_;
  int        generation_;

  FieldIPort* ifield_;
  FieldHandle ifieldH_;

  ITKDatatypeIPort* iimg_;
  ITKDatatypeHandle iimgH_;

  NrrdOPort* onrrd_;
  NrrdDataHandle onrrdH_;
  
public:
  BuildSeedVolume(GuiContext*);
  
  virtual ~BuildSeedVolume();
  
  virtual void execute();
  
};
  
  
DECLARE_MAKER(BuildSeedVolume)
BuildSeedVolume::BuildSeedVolume(GuiContext* ctx)
  : Module("BuildSeedVolume", ctx, Source, "Converters", "Insight"),
    gui_inside_value_(ctx->subVar("inside_value")),
    gui_outside_value_(ctx->subVar("outside_value")),
    generation_(-1)
{
}
  
BuildSeedVolume::~BuildSeedVolume(){
}
  
void
BuildSeedVolume::execute()
{
  // check input ports
  ifield_ = (FieldIPort *)get_iport("InputField");

  if (!ifield_)
  {
    error( "Unable to initialize iport 'InputField'.");
    return;
  }

  if (!(ifield_->get(ifieldH_) && ifieldH_.get_rep()))
  {
    error( "No handle or representation in input field." );
    return;
  }

  iimg_ = (ITKDatatypeIPort *)get_iport("InputImage");

  if (!iimg_)
  {
    error( "Unable to initialize iport 'InputImage'.");
    return;
  }

  if (!(iimg_->get(iimgH_) && iimgH_.get_rep()))
  {
    error( "No handle or representation in input image." );
    return;
  }
  
  // check output ports
  onrrd_ = (NrrdOPort *)get_oport("OutputImage");
  if(!onrrd_) {
    error("Unable to initialize oport");
    return;
  }

  // input is a PointCloudField<double>
  typedef PointCloudField<double> FieldType;

  if (!dynamic_cast<FieldType*>(ifieldH_.get_rep())) {
    error("BuildSeedVolume only supports PointsClouds of doubles.");
    return;
  }

  itk::MultiThreader::SetGlobalMaximumNumberOfThreads(1);
  
  FieldType *f = dynamic_cast<FieldType*>(ifieldH_.get_rep());


  // input image must be of dimension 2
  //FIX ME: make it work for various image type
  typedef itk::Image<float, 2> InputImageType;

  if (!dynamic_cast<InputImageType*>(iimgH_.get_rep()->data_.GetPointer())) {
    error("BuildSeedVolume only supports 2D input images of type float");
    return;
  }

  InputImageType *in_img = dynamic_cast<InputImageType*>(iimgH_.get_rep()->data_.GetPointer());

  // query input image info for size, spacing, min points
  int samples_x=0, samples_y=0;
  double spacing_x=1, spacing_y=1;
  double min_x=0.0, min_y=0.0;
  samples_x = (in_img->GetLargestPossibleRegion()).GetSize()[0];
  samples_y = (in_img->GetLargestPossibleRegion()).GetSize()[1];

  spacing_x = in_img->GetSpacing()[0];
  spacing_y = in_img->GetSpacing()[1];

  min_x = in_img->GetOrigin()[0];
  min_y = in_img->GetOrigin()[1];

  // create nrrd
  NrrdData *n = scinew NrrdData();
  nrrdAlloc(n->nrrd, nrrdTypeFloat, 2, samples_x, samples_y);
  n->nrrd->axis[0].spacing = spacing_x;
  n->nrrd->axis[1].spacing = spacing_y;
  n->nrrd->axis[0].min = min_x;
  n->nrrd->axis[1].min = min_y;

  // iterate over each node with position x,y,z
  FieldType::mesh_handle_type imesh = f->get_typed_mesh();
  PointCloudMesh::Node::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);
  float *data = (float*)n->nrrd->data;
  float inside = (float)gui_inside_value_.get();
  float outside = (float)gui_outside_value_.get();

  // set all values in nrrd to outside value
  for (int i=0; i<samples_x*samples_y; i++)
    data[i] = outside;

  // for each node, set pixel to inside_value for all pixels 
  // within the node's given radius
  double rad;
  double x_rad, y_rad;
  while (ibi != iei)
  {
    Point p;
    imesh->get_center(p, *ibi);

    int x = (int)p.x();
    int y = (int)p.y();
    f->value(rad, *ibi);

    // convert to image space
    x = x / spacing_x;
    y = y / spacing_y;
    x_rad = rad / spacing_x;
    y_rad = rad / spacing_y;

    // fill in that pixel
    int radsq = x_rad * y_rad;
    for(int r=y-y_rad; r <=(y+y_rad); r++) {
      for(int c=x-x_rad; c <= (x+x_rad); c++) {
	// check if point is within volume and
	// should be part of the circle
	// if so fill it in
	if (r>0 && r<samples_y && c>0 && c<samples_x &&
	    ((c-x)*(c-x) + (r-y)*(r-y)) <= radsq) 
	  data[r*samples_x+c+1] = inside;
      }
    }
    ++ibi;
  }

  // send output image

  onrrdH_ = n;
  onrrd_->send(onrrdH_);
}

  
}
  
