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
//    File   : ITKThresholdSegmentationLevelSetImageFilterTool.h
//    Author : McKay Davis
//    Date   : Sat Oct 14 14:50:14 2006


#ifndef SEG3D_ITKThresholdSegmentationLevelSetImageFilterTool_h
#define SEG3D_ITKThresholdSegmentationLevelSetImageFilterTool_h

#include <sci_defs/insight_defs.h>

#ifdef HAVE_INSIGHT
#include <string>
#include <StandAlone/Apps/Seg3D/NrrdVolume.h>
#include <StandAlone/Apps/Seg3D/VolumeFilter.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Datatypes/ITKDatatype.h>
#include <itkImageToImageFilter.h>
#include <itkCommand.h>
#include <itkThresholdSegmentationLevelSetImageFilter.h>

namespace SCIRun {

class Painter;
typedef itk::Image<float,3> ITKImageFloat3D;

class ITKThresholdSegmentationLevelSetImageFilterTool : public BaseTool {
public:
  ITKThresholdSegmentationLevelSetImageFilterTool(Painter *painter);
  propagation_state_e           process_event(event_handle_t);
private:
  void                          finish();
  void                          cont();
  void                          set_vars();
  Painter *                     painter_;
  NrrdVolumeHandle              seed_volume_;
  Skinner::Var<double>          LowerThreshold_;
  Skinner::Var<double>          UpperThreshold_;

  
  typedef itk::ThresholdSegmentationLevelSetImageFilter
  < ITKImageFloat3D, ITKImageFloat3D > FilterType;
  typedef FilterType::FeatureImageType FeatureImg;
  
  VolumeFilter<FilterType>      filter_;
};

}

#endif // HAVE_INSIGHT

#endif
