//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
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
//    File   : ITKFilterCallback.h
//    Author : McKay Davis
//    Date   : Sun Nov  5 21:37:09 2006

#ifndef LeXoV_ITKFilterCallback_H
#define LeXoV_ITKFilterCallback_H

#include <sci_comp_warn_fixes.h>
#include <StandAlone/Apps/Painter/Painter.h>
#include <StandAlone/Apps/Painter/NrrdVolume.h>
#include <StandAlone/Apps/Painter/NrrdToITK.h>
#include <Core/Skinner/Parent.h>
#include <Core/Skinner/Variables.h>
#include <Core/Util/Assert.h>

#include <sci_defs/insight_defs.h>

#ifdef HAVE_INSIGHT
#  include <Core/Datatypes/ITKDatatype.h>
#  include <itkCommand.h>
#endif

namespace SCIRun {

#ifdef HAVE_INSIGHT
using SCIRun::ITKDatatypeHandle;

// todo, integrate scirun progressreporter class to this callback
template <class FilterType>
class ITKFilterCallback : public Skinner::Parent
{
  typedef typename FilterType::Pointer FilterPointer;
public:
  ITKFilterCallback (Skinner::Variables *vars,
                     NrrdVolumeHandle volume,
                     FilterPointer filter) :
    Parent::Parent(vars),
    filter_(filter),
    volume_(volume),
    abort_(false),
    progress_(0.0),
    passes_(1)
  {
  }
  
  
  virtual ~ITKFilterCallback() {};
  void                  operator()(NrrdDataHandle nrrdh = 0);
    
private:
  void                  filter_callback(itk::Object *, 
                                        const itk::EventObject &);
  
  void                  filter_callback_const (const itk::Object *, 
                                               const itk::EventObject &);

  FilterPointer         filter_;
  NrrdVolumeHandle      volume_;
  bool                  abort_;
  double                progress_;
  int                   passes_;
  
};
  

template <class FilterType>  
void
ITKFilterCallback<FilterType>::operator()(NrrdDataHandle nrrd_handle) 
{
  if (!nrrd_handle.get_rep()) {
    nrrd_handle = volume_->nrrd_handle_;
  }
  typedef typename FilterType::InputImageType ImageInT;
  typedef typename FilterType::OutputImageType ImageOutT;
  typedef typename 
    itk::MemberCommand< ITKFilterCallback<FilterType> > RedrawCommandType;

  typename RedrawCommandType::Pointer callback = RedrawCommandType::New();

  callback->SetCallbackFunction
    (this, &ITKFilterCallback<FilterType>::filter_callback);

  callback->SetCallbackFunction
    (this, &ITKFilterCallback<FilterType>::filter_callback_const);

  filter_->AddObserver(itk::ProgressEvent(), callback);
  filter_->AddObserver(itk::IterationEvent(), callback);
  
  if (nrrd_handle.get_rep()) {
    ITKDatatypeHandle img_handle = nrrd_to_itk_image(nrrd_handle);  
    ImageInT *imgp = dynamic_cast<ImageInT*>(img_handle->data_.GetPointer());
    if (!imgp) {
      return;
    }    
    filter_->SetInput(imgp);
  }

  try {
    filter_->Update();
  } catch (itk::ExceptionObject &err) {
    if (!abort_) {
      cerr << "ITK Exception: \n";
      err.Print(cerr);
      return;
    } else {
      abort_ = false;
    }
  } catch (...) {
    cerr << "ITK Filter error!\n";
    return;
  }
  
  SCIRun::ITKDatatypeHandle output_img = new SCIRun::ITKDatatype();
  output_img->data_ = filter_->GetOutput();
  volume_->nrrd_handle_ = 
    itk_image_to_nrrd<typename ImageOutT::PixelType>(output_img);
}


template <class FilterType>
void
ITKFilterCallback<FilterType>::filter_callback(itk::Object *object,
                                               const itk::EventObject &event)
{
  typedef typename FilterType::InputImageType ImageInT;
  typedef typename FilterType::OutputImageType ImageOutT;
  typedef itk::ProcessObject::Pointer ProcessPointer;

  ProcessPointer process = dynamic_cast<itk::ProcessObject *>(object);
  ASSERT(process);
  if (!process) return;

  progress_ =  process->GetProgress();

  if (typeid(itk::ProgressEvent) == typeid(event))
  {
    std::cerr << "Filter progress: " << progress_ * 100.0 << "%\n";
    if (volume_.get_rep()) {
      ITKDatatypeHandle imgh = new ITKDatatype();
      imgh->data_ = filter_->GetOutput();

      if (!dynamic_cast<ImageOutT *>(imgh->data_.GetPointer())) return;

      typedef typename FilterType::OutputImageType::PixelType OutT;
      volume_->nrrd_handle_ = itk_image_to_nrrd<OutT>(imgh);
      volume_->dirty_ = true;
      volume_->painter_->extract_all_window_slices();
      volume_->painter_->redraw_all();
    }
  }


  if (typeid(itk::IterationEvent) == typeid(event))
  {
    std::cerr << "Filter Iteration: " << progress_ * 100.0 << "%\n";
  }

  if (abort_) {
    process->AbortGenerateDataOn();
  }
}


template <class FilterType>
void
ITKFilterCallback<FilterType>::
filter_callback_const(const itk::Object *, const itk::EventObject &)
{
  cerr << "filter_callback_const not implemented\n;";
}

#endif



}
#endif
