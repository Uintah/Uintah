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
//    File   : PainterSignalTargets.cc
//    Author : McKay Davis
//    Date   : Mon Jul  3 21:47:22 2006


#include <StandAlone/Apps/Painter/Painter.h>
#include <sci_comp_warn_fixes.h>
#include <iostream>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <Core/Bundle/Bundle.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/TkOpenGLContext.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/GuiInterface/UIvar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/CleanupManager.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Environment.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Geom/FontManager.h>
#include <Core/Util/SimpleProfiler.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Util/FileUtils.h>

#ifdef HAVE_INSIGHT

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#endif

namespace SCIRun {


BaseTool::propagation_state_e 
Painter::InitializeSignalCatcherTargets(event_handle_t) {
  REGISTER_CATCHER_TARGET(Painter::Autoview);
  REGISTER_CATCHER_TARGET(Painter::quit);
  REGISTER_CATCHER_TARGET(Painter::SliceWindow_Maker);
  REGISTER_CATCHER_TARGET(Painter::StartITKGradientTool);
  REGISTER_CATCHER_TARGET(Painter::ITKImageFileRead);
  REGISTER_CATCHER_TARGET(Painter::ITKImageFileWrite);
  REGISTER_CATCHER_TARGET(Painter::NrrdFileRead);
  REGISTER_CATCHER_TARGET(Painter::CopyLayer);
  REGISTER_CATCHER_TARGET(Painter::DeleteLayer);
  REGISTER_CATCHER_TARGET(Painter::NewLayer);
  REGISTER_CATCHER_TARGET(Painter::StartBrushTool);
  return STOP_E;
}




BaseTool::propagation_state_e 
Painter::StartBrushTool(event_handle_t event) {
  cerr << "Painter::start_brush_tool";
  tm_.add_tool(new BrushTool(this),25); 
  return STOP_E;
}

BaseTool::propagation_state_e 
Painter::quit(event_handle_t event) {
  EventManager::add_event(new QuitEvent());
  return STOP_E;
}

BaseTool::propagation_state_e 
Painter::CopyLayer(event_handle_t) {
  copy_current_layer();
  return STOP_E;
}

BaseTool::propagation_state_e 
Painter::DeleteLayer(event_handle_t) {
  kill_current_layer();
  return STOP_E;
}

BaseTool::propagation_state_e 
Painter::NewLayer(event_handle_t) {
  new_current_layer();
  return STOP_E;
}


BaseTool::propagation_state_e 
Painter::NrrdFileRead(event_handle_t event) {
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  const string &filename = signal->get_signal_data();
  if (!validFile(filename)) {
    return STOP_E;
  }

  NrrdDataHandle nrrd_handle = new NrrdData();
  Nrrd *nrrd = nrrd_handle->nrrd_;
  if (nrrdLoad(nrrd, filename.c_str(), 0)) {
    get_vars()->insert("Painter::status_text",
                       "Cannot Load Nrrd: "+filename, 
                       "string", true);
    
  } else {
    BundleHandle bundle = new Bundle();
    bundle->setNrrd(filename, nrrd_handle);
    add_bundle(bundle); 
    get_vars()->insert("Painter::status_text",
                       "Successfully Loaded Nrrd: "+filename,
                       "string", true);

  }
  return STOP_E;  
}


BaseTool::propagation_state_e 
Painter::Autoview(event_handle_t) {
  if (current_volume_) {
    SliceWindows::iterator window = windows_.begin();
    SliceWindows::iterator end = windows_.end();
    for (;window != end; ++window) {
      (*window)->autoview(current_volume_);
    }
  }

  return STOP_E;
}



BaseTool::propagation_state_e 
Painter::StartITKGradientTool(event_handle_t) {
#ifdef HAVE_INSIGHT
  tm_.add_tool(new ITKGradientMagnitudeTool(this),100); 
#endif
  return STOP_E;
}



BaseTool::propagation_state_e 
Painter::SliceWindow_Maker(event_handle_t event) {
  //  event.detach();
  Skinner::MakerSignal *maker_signal = 
    dynamic_cast<Skinner::MakerSignal *>(event.get_rep());
  ASSERT(maker_signal);

  SliceWindow *window = new SliceWindow(maker_signal->get_vars(), this);
  windows_.push_back(window);
  maker_signal->set_signal_thrower(window);
  maker_signal->set_signal_name(maker_signal->get_signal_name()+"_Done");
  return MODIFIED_E;
}



BaseTool::propagation_state_e 
Painter::ITKImageFileRead(event_handle_t event) {
#ifdef HAVE_INSIGHT
  cerr << "ITKImageFileRead\n";
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);

  const string &filename = signal->get_signal_data();
  if (!validFile(filename)) return STOP_E;

  typedef itk::ImageFileReader<itk::Image<float, 3> > FileReaderType;
  
  // create a new reader
  FileReaderType::Pointer reader = FileReaderType::New();
  
  // set reader
  reader->SetFileName( filename.c_str() );
  
  try {
    reader->Update();  
  } catch  ( itk::ExceptionObject & err ) {
    cerr << ("ExceptionObject caught!");
    cerr << (err.GetDescription());
  }
  

  Insight::ITKDatatype *img = new Insight::ITKDatatype();
  img->data_ = reader->GetOutput();

  if (!img->data_) { 
    cerr << "no itk image\n";
  }

  ITKDatatypeHandle img_handle = img;
  NrrdDataHandle nrrd_handle = itk_image_to_nrrd(img_handle);

  if (nrrd_handle->nrrd_) {
    cerr << "nrrd converted!\n";
    BundleHandle bundle = new Bundle(); 
    bundle->setNrrd(filename, nrrd_handle);
    add_bundle(bundle); 
  }

  // ITKDataTypeSignal *return_event = new ITKDataTypeSignal(img);
  //  event = return_event;
#endif
  return MODIFIED_E;
}



BaseTool::propagation_state_e 
Painter::ITKImageFileWrite(event_handle_t event) {
#ifdef HAVE_INSIGHT
  cerr << "ITKImageFileWrite\n";

  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);

  const string &filename = signal->get_signal_data();

  typedef itk::ImageFileWriter<itk::Image<float, 3> > FileWriterType;
  
  // create a new writer
  FileWriterType::Pointer writer = FileWriterType::New();
  
  ITKDatatypeHandle itk_image_h = nrrd_to_itk_image(current_volume_->nrrd_handle_);
  typedef FileWriterType::InputImageType ImageType;
  ImageType *img = dynamic_cast<ImageType *>(itk_image_h->data_.GetPointer());
  ASSERT(img);


  // set writer
  writer->SetFileName( filename.c_str() );
  writer->SetInput(img);
  
  try {
    writer->Update();  
    cerr << "ITKImageFileWrite success";
  } catch  ( itk::ExceptionObject & err ) {
    cerr << ("ExceptionObject caught!");
    cerr << (err.GetDescription());
  }
  

  // ITKDataTypeSignal *return_event = new ITKDataTypeSignal(img);
  //  event = return_event;
#endif
  return MODIFIED_E;
}


  
} // end namespace SCIRun
