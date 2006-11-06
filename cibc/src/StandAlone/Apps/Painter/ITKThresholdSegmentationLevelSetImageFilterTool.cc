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
#include <StandAlone/Apps/Painter/VolumeOps.h>
#include <StandAlone/Apps/Painter/ITKFilterCallback.h>
#include <Core/Util/Assert.h>

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

    if (!seed_volume_.get_rep()) {
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
    finish();
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
  painter_->do_itk_filter<FilterType>(filter_, temp);
}


#define SetFilterVarMacro(name, type) \
  filter_->Set##name(painter_->get_vars()->get_##type(scope+#name));



void
ITKThresholdSegmentationLevelSetImageFilterTool::set_vars()
{
  string scope = "ITKThresholdSegmentationLevelSetImageFilterTool::";
  SetFilterVarMacro(CurvatureScaling, double);
  SetFilterVarMacro(PropagationScaling, double);
  SetFilterVarMacro(EdgeWeight, double);
  SetFilterVarMacro(NumberOfIterations, int);
  SetFilterVarMacro(MaximumRMSError, double);
  SetFilterVarMacro(IsoSurfaceValue, double);
  SetFilterVarMacro(SmoothingIterations,int);
  SetFilterVarMacro(SmoothingTimeStep, double);
  SetFilterVarMacro(SmoothingConductance, double);
  if (painter_->get_vars()->get_bool(scope+"ReverseExpansionDirection")) 
    filter_->ReverseExpansionDirectionOn();
  else 
    filter_->ReverseExpansionDirectionOff();
}
  



void
ITKThresholdSegmentationLevelSetImageFilterTool::finish()
{
  painter_->volume_lock_.lock();
  NrrdVolumeHandle &vol = painter_->current_volume_;

  string newname = 
    painter_->unique_layer_name(vol->name_+" ITK Threshold Result");
  NrrdDataHandle extracted_nrrd = 
    VolumeOps::bit_to_float(seed_volume_->nrrd_handle_, 
                            seed_volume_->label_, 1000.0);

  NrrdDataHandle clear_volume = 
    VolumeOps::create_clear_nrrd(extracted_nrrd, nrrdTypeFloat);
  
  NrrdVolume *new_layer = 
    new NrrdVolume(painter_, newname, extracted_nrrd, 2);

  painter_->volumes_.push_back(new_layer);
  painter_->rebuild_layer_buttons();
  painter_->extract_all_window_slices();
  
  pair<double, double> mean = painter_->compute_mean_and_deviation
    (painter_->current_volume_->nrrd_handle_->nrrd_, 
     new_layer->nrrd_handle_->nrrd_);
  painter_->volume_lock_.unlock();

  double factor = 2.5;
  double min = mean.first - factor*mean.second;
  double max = mean.first + factor*mean.second;
  painter_->status_ = ("Threshold min: " + to_string(min) + 
                       " Threshold max: " + to_string(max));


  filter_ = FilterType::New();
  filter_->SetLowerThreshold(min);
  filter_->SetUpperThreshold(max);
  
  set_vars();

  filter_->SetFeatureImage
    (dynamic_cast<ITKImageFloat3D *>
     (painter_->current_volume_->get_itk_image()->data_.GetPointer()));

  ITKFilterCallback<FilterType>(painter_->get_vars(),new_layer,filter_)();

  new_layer->change_type_from_float_to_bit();

  painter_->redraw_all();
}


}

#endif
