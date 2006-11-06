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
//    File   : ITKConfidenceConnectedImageFilterTool.h
//    Author : McKay Davis
//    Date   : Sat Oct 14 15:51:56 2006

#ifndef LEXOV_ITKConfidenceConnectedimageFilterTool_h
#define LEXOV_ITKConfidenceConnectedimageFilterTool_h

#include <sci_defs/insight_defs.h>

#ifdef HAVE_INSIGHT
#include <string>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Datatypes/ITKDatatype.h>
#include <itkImageToImageFilter.h>
#include <itkCommand.h>
#include <itkThresholdSegmentationLevelSetImageFilter.h>

namespace SCIRun {

class Painter;
class NrrdVolume;
typedef itk::Image<float,3> ITKImageFloat3D;

  
class ITKConfidenceConnectedImageFilterTool : public virtual BaseTool,
                                              public PointerTool
{
public:
  ITKConfidenceConnectedImageFilterTool(Painter *painter);
  propagation_state_e   pointer_down(int, int, int, unsigned int, int);
  propagation_state_e   pointer_up(int, int, int, unsigned int, int);
  propagation_state_e   pointer_motion(int, int, int, unsigned int, int);
  propagation_state_e   process_event(event_handle_t);
private:
  void                  draw_gl(SliceWindow &window);
  void                  finish();
  Painter *             painter_;
  vector<int>           seed_;
  NrrdVolumeHandle      volume_;
  string                prefix_;
  Skinner::Var<int>     numberOfIterations_;
  Skinner::Var<double>  multiplier_;
  Skinner::Var<double>  replaceValue_;
  Skinner::Var<int>     initialNeighborhoodRadius_;
};
  
  
}

#endif // HAVE_INSIGHT

#endif
