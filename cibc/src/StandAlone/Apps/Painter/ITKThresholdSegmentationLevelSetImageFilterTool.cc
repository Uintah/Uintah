//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ITKThresholdSegmentationLevelSetImageFilterTool.cc
//    Author : McKay Davis
//    Date   : Sat Oct 14 14:52:24 2006

#include <StandAlone/Apps/Painter/ITKThresholdSegmentationLevelSetImageFilterTool.h>
#ifdef HAVE_INSIGHT
#include <StandAlone/Apps/Painter/Painter.h>

namespace SCIRun {

ITKThresholdSegmentationLevelSetImageFilterTool::ITKThresholdSegmentationLevelSetImageFilterTool(Painter *painter) :
  BaseTool("ITK Threshold"),
  painter_(painter),
  seed_volume_(0),
  filter_(0)
{
}

BaseTool::propagation_state_e 
ITKThresholdSegmentationLevelSetImageFilterTool::process_event
(event_handle_t event)
{

  if (dynamic_cast<SetLayerEvent *>(event.get_rep())) {
    seed_volume_ = painter_->current_volume_;
    painter_->redraw_all();
  }

  if (dynamic_cast<FinishEvent *>(event.get_rep())) {
    if (painter_->current_volume_  == seed_volume_) {
      painter_->status_ = "Cannot use same layers for source and seed";
      painter_->redraw_all();
      return STOP_E;
    }

    if (!seed_volume_) {
      painter_->status_ = "No seed layer set";
      painter_->redraw_all();
      return STOP_E;
    }

    if (filter_.IsNull()) 
      finish();
    else 
      cont();
    return CONTINUE_E;
  }

  if (dynamic_cast<QuitEvent *>(event.get_rep())) {
    return QUIT_AND_CONTINUE_E;
  }
 
  return CONTINUE_E;
}

void
ITKThresholdSegmentationLevelSetImageFilterTool::cont()
{
  cerr << "CONT\n";
  //  filter_->ReverseExpansionDirectionOff();
  filter_->ManualReinitializationOn();
  filter_->Modified();
  NrrdDataHandle temp = 0;
  set_vars();
  painter_->do_itk_filter<ITKImageFloat3D>(filter_, temp);
}


void
ITKThresholdSegmentationLevelSetImageFilterTool::set_vars()
{
  ASSERT(filter_);
  string scope = "ITKThresholdSegmentationLevelSetImageFilterTool::";
  Skinner::Variables *vars = painter_->get_vars();
  filter_->SetCurvatureScaling(vars->get_double(scope+"curvatureScaling"));
  filter_->SetPropagationScaling(vars->get_double(scope+"propagationScaling"));
  filter_->SetEdgeWeight(vars->get_double(scope+"edgeWeight"));
  filter_->SetNumberOfIterations(vars->get_int(scope+"numberOfIterations"));
  filter_->SetMaximumRMSError(vars->get_double(scope+"maximumRMSError"));
  if (vars->get_bool(scope+"reverseExpansionDirection")) 
    filter_->ReverseExpansionDirectionOn();
  else 
    filter_->ReverseExpansionDirectionOff();
  filter_->SetIsoSurfaceValue(vars->get_double(scope+"isoSurfaceValue"));
  filter_->SetSmoothingIterations(vars->get_int(scope+"smoothingIterations"));
  filter_->SetSmoothingTimeStep(vars->get_double(scope+"smoothingTimeStep"));
  filter_->SetSmoothingConductance(vars->get_double(scope+"smoothingConductance"));
}
  



void
ITKThresholdSegmentationLevelSetImageFilterTool::finish()
{
  NrrdDataHandle source_nrrdh = painter_->current_volume_->nrrd_handle_;
  filter_ = FilterType::New();

  string name = "ITK Threshold Result";
  NrrdVolume *new_layer = new NrrdVolume(seed_volume_, name, 0);
  new_layer->colormap_ = 1;
  new_layer->data_min_ = -4.0;
  new_layer->data_max_ = 4.0;
  new_layer->clut_min_ = 4.0/255.0;
  new_layer->clut_max_ = 4.0;

  //  new_layer->clut_min_ = -4.0;//0.5/255.0;
  //  new_layer->clut_max_ = 4.0;//0.5;
  painter_->volumes_.push_back(new_layer);
  
  NrrdDataHandle seed_nrrdh = new_layer->nrrd_handle_;

  name = "ITK Threshold Seed";
  pair<double, double> mean = painter_->compute_mean_and_deviation
    (source_nrrdh->nrrd_, seed_nrrdh->nrrd_);

  double factor = 2.5;
  double min = mean.first - factor*mean.second;
  double max = mean.first + factor*mean.second;


  filter_->SetLowerThreshold(min);
  filter_->SetUpperThreshold(max);


  painter_->status_ = ("Threshold min: " + to_string(min) + 
                       " Threshold max: " + to_string(max));
  
  set_vars();

  ITKDatatypeHandle img_handle = nrrd_to_itk_image(source_nrrdh);
  ITKImageFloat3D *imgp = 
    dynamic_cast<ITKImageFloat3D *>(img_handle->data_.GetPointer());
  ASSERT(imgp);
  filter_->SetFeatureImage(imgp);

  painter_->filter_volume_ = new_layer;
  painter_->filter_update_img_ = nrrd_to_itk_image(seed_nrrdh);

  painter_->do_itk_filter<ITKImageFloat3D>(filter_, seed_nrrdh);
  new_layer->nrrd_handle_ = seed_nrrdh;

  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
}


}

#endif
