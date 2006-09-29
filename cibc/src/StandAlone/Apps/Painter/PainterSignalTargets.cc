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
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
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
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Volume/ColorMap2.h>
#include <Core/Volume/VolumeRenderer.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>

#ifndef _WIN32
#include <sys/mman.h>
#endif

#ifdef HAVE_INSIGHT

#  include <itkImageFileReader.h>
#  include <itkImageFileWriter.h>
#  include <itkGradientMagnitudeImageFilter.h>
#  include <itkConfidenceConnectedImageFilter.h>
#  include <itkCurvatureAnisotropicDiffusionImageFilter.h>
#  include <itkBinaryBallStructuringElement.h>
#  include <itkBinaryDilateImageFilter.h>
#  include <itkBinaryErodeImageFilter.h>
#  include <itkImportImageFilter.h>

#endif

namespace SCIRun {


BaseTool::propagation_state_e 
Painter::InitializeSignalCatcherTargets(event_handle_t) {
  REGISTER_CATCHER_TARGET(Painter::SliceWindow_Maker);
  REGISTER_CATCHER_TARGET(Painter::LayerButton_Maker);

  REGISTER_CATCHER_TARGET(Painter::StartBrushTool);
  REGISTER_CATCHER_TARGET(Painter::StartCropTool);
  REGISTER_CATCHER_TARGET(Painter::StartFloodFillTool);

  REGISTER_CATCHER_TARGET(Painter::Autoview);
  REGISTER_CATCHER_TARGET(Painter::CopyLayer);
  REGISTER_CATCHER_TARGET(Painter::DeleteLayer);
  REGISTER_CATCHER_TARGET(Painter::NewLayer);
  REGISTER_CATCHER_TARGET(Painter::MergeLayer);

  REGISTER_CATCHER_TARGET(Painter::MemMapFileRead);
  REGISTER_CATCHER_TARGET(Painter::NrrdFileRead);
  REGISTER_CATCHER_TARGET(Painter::NrrdFileWrite);

  REGISTER_CATCHER_TARGET(Painter::CancelTool);  
  REGISTER_CATCHER_TARGET(Painter::FinishTool);
  REGISTER_CATCHER_TARGET(Painter::SetLayer);
  REGISTER_CATCHER_TARGET(Painter::LoadColorMap1D);

  REGISTER_CATCHER_TARGET(Painter::ITKBinaryDilate);  
  REGISTER_CATCHER_TARGET(Painter::ITKImageFileRead);
  REGISTER_CATCHER_TARGET(Painter::ITKImageFileWrite);
  REGISTER_CATCHER_TARGET(Painter::ITKGradientMagnitude);
  REGISTER_CATCHER_TARGET(Painter::ITKBinaryDilateErode);
  REGISTER_CATCHER_TARGET(Painter::ITKCurvatureAnisotropic);
  REGISTER_CATCHER_TARGET(Painter::ITKConfidenceConnected);
  REGISTER_CATCHER_TARGET(Painter::ITKThresholdLevelSet);

  REGISTER_CATCHER_TARGET(Painter::ShowVolumeRendering);

  REGISTER_CATCHER_TARGET(Painter::AbortFilterOn);

  REGISTER_CATCHER_TARGET(Painter::ResampleVolume);
   
  return STOP_E;
}


BaseTool::propagation_state_e 
Painter::SliceWindow_Maker(event_handle_t event) {
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
Painter::LayerButton_Maker(event_handle_t event) {
  Skinner::MakerSignal *maker_signal = 
    dynamic_cast<Skinner::MakerSignal *>(event.get_rep());
  ASSERT(maker_signal);

  LayerButton *lb = new LayerButton(maker_signal->get_vars(), this);
  layer_buttons_.push_back(lb);
  maker_signal->set_signal_thrower(lb);
  maker_signal->set_signal_name(maker_signal->get_signal_name()+"_Done");
  return MODIFIED_E;
}



BaseTool::propagation_state_e 
Painter::StartBrushTool(event_handle_t event) {
  tm_.add_tool(new BrushTool(this),25); 
  return CONTINUE_E;
}
  

BaseTool::propagation_state_e 
Painter::StartCropTool(event_handle_t event) {
#if 0
  get_vars()->insert("ToolDialog::text", " Crop Volume...",
                     "string", true);
  get_vars()->insert("Painter::progress_bar_text", "", "string", true);
  get_vars()->insert("ProgressBar::bar_height","0","string",true);
  get_vars()->insert("Painter::progress_bar_total_width","0","string", true);
  get_vars()->unset("ToolDialog::button_height"); 
#endif
  redraw_all();

  tm_.add_tool(new CropTool(this),25);
  redraw_all();
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::StartFloodFillTool(event_handle_t event) {
  tm_.add_tool(new FloodfillTool(this),25);
  return CONTINUE_E;
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

  return CONTINUE_E;
}




BaseTool::propagation_state_e 
Painter::CopyLayer(event_handle_t) {
  copy_current_layer();
  return CONTINUE_E;
}

BaseTool::propagation_state_e 
Painter::DeleteLayer(event_handle_t) {
  kill_current_layer();
  return CONTINUE_E;
}

BaseTool::propagation_state_e 
Painter::NewLayer(event_handle_t) {
  new_current_layer();
  return CONTINUE_E;
}



BaseTool::propagation_state_e 
Painter::MergeLayer(event_handle_t event) {
  NrrdVolumeOrder::iterator volname = 
    std::find(volume_order_.begin(), 
              volume_order_.end(), 
              current_volume_->name_);
  
  if (volname == volume_order_.begin()) return STOP_E;
  NrrdVolume *vol1 = volume_map_[*volname];
  NrrdVolume *vol2 = volume_map_[*(--volname)];
    

  NrrdData *nout = new NrrdData();
  NrrdIter *ni1 = nrrdIterNew();
  NrrdIter *ni2 = nrrdIterNew();
    
  nrrdIterSetNrrd(ni1, vol1->nrrd_handle_->nrrd_);
  nrrdIterSetNrrd(ni2, vol2->nrrd_handle_->nrrd_);
  
  if (nrrdArithIterBinaryOp(nout->nrrd_, nrrdBinaryOpMultiply, ni1, ni2)) {
    char *err = biffGetDone(NRRD);
    string errstr = (err ? err : "");
    free(err);
    throw errstr;
  }

  nrrdIterNix(ni1);
  nrrdIterNix(ni2);

  nrrdKeyValueCopy(nout->nrrd_,  vol1->nrrd_handle_->nrrd_);
  nrrdKeyValueCopy(nout->nrrd_,  vol2->nrrd_handle_->nrrd_);
  
  vol1->nrrd_handle_->nrrd_ = nout->nrrd_;
  vol2->keep_ = 0;
  
  recompute_volume_list();
  current_volume_ = vol1;
  return CONTINUE_E;
}
  

BaseTool::propagation_state_e 
Painter::NrrdFileRead(event_handle_t event) {
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  const string &filename = signal->get_vars()->get_string("filename");
  if (!validFile(filename)) {
    return STOP_E;
  }

  NrrdDataHandle nrrd_handle = new NrrdData();
  Nrrd *nrrd = nrrd_handle->nrrd_;

  if (nrrdLoad(nrrd, filename.c_str(), 0)) {
    get_vars()->insert("Painter::status_text",
                       "Cannot Load Nrrd: "+filename, 
                       "string", true);
    return STOP_E;
    
  } 
  
  pair<string, string> dirfile = split_filename(filename);
  BundleHandle bundle = new Bundle();
  bundle->setNrrd(dirfile.second, nrrd_handle);
  add_bundle(bundle); 
  get_vars()->insert("Painter::status_text",
                     "Successfully Loaded Nrrd: "+filename,
                     "string", true);

  return CONTINUE_E;  
}


BaseTool::propagation_state_e 
Painter::MemMapFileRead(event_handle_t event) {
#ifdef _WIN32
  // no mmap on windows
  cerr << "  MMap not available\n";
  return STOP_E;
#endif

  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  const string &filename = signal->get_vars()->get_string("filename");
  //  if (!validFile(filename)) {
  //    return STOP_E;
  //  }
  const string datafile = filename+".img";
  const string nrrdfile = filename+".nhdr";

  NrrdDataHandle nrrd_handle = new NrrdData();
  Nrrd *nrrd = nrrd_handle->nrrd_;

  int fd = open(datafile.c_str(), O_RDONLY);
  if (!fd) {
    cerr << "Not opened\n";
    return STOP_E;
  }


  if (nrrdLoad(nrrd, nrrdfile.c_str(), 0)) {
    get_vars()->insert("Painter::status_text",
                       "Cannot Load Nrrd: "+filename, 
                       "string", true);
    //    return STOP_E;
    
  } 

  struct stat buf;
  fstat(fd, &buf);
#ifndef _WIN32
  nrrd->data = mmap(0, buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
#endif
  pair<string, string> dirfile = split_filename(filename);
  BundleHandle bundle = new Bundle();
  bundle->setNrrd(dirfile.second, nrrd_handle);
  add_bundle(bundle); 
  get_vars()->insert("Painter::status_text",
                     "Successfully Loaded Nrrd: "+filename,
                     "string", true);

  return CONTINUE_E;  
}

BaseTool::propagation_state_e 
Painter::NrrdFileWrite(event_handle_t event) {
  ASSERTMSG(0, "Not implemented");
  return STOP_E;  
}


BaseTool::propagation_state_e 
Painter::ITKBinaryDilate(event_handle_t event) {
#ifdef HAVE_INSIGHT
  string name = "ITKBinaryDilate";
  typedef itk::BinaryBallStructuringElement< float, 3> StructuringElementType;
  typedef itk::BinaryDilateImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D, StructuringElementType > FilterType;
  FilterType::Pointer filter = FilterType::New();

  StructuringElementType structuringElement;
  structuringElement.SetRadius
    (get_vars()->get_int(name+"::radius"));
  structuringElement.CreateStructuringElement();
  
  filter->SetKernel(structuringElement);
  filter->SetDilateValue
    (get_vars()->get_double(name+"::dilateValue"));

  NrrdVolume *vol = current_volume_;
  do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);
  redraw_all();
#endif
  return CONTINUE_E;
}




BaseTool::propagation_state_e 
Painter::ITKImageFileRead(event_handle_t event) {

#ifndef HAVE_INSIGHT
  return NrrdFileRead(event);
#else

  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);

  const string &filename = signal->get_vars()->get_string("filename");
  if (!validFile(filename)) {
    return STOP_E;
  }

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
  

  SCIRun::ITKDatatype *img = new SCIRun::ITKDatatype();
  img->data_ = reader->GetOutput();

  if (!img->data_) { 
    cerr << "no itk image\n";
    return STOP_E;
  }

  ITKDatatypeHandle img_handle = img;
  NrrdDataHandle nrrd_handle = itk_image_to_nrrd(img_handle);

  pair<string, string> dirfile = split_filename(filename);

  if (nrrd_handle->nrrd_) {
    BundleHandle bundle = new Bundle(); 
    bundle->setNrrd(dirfile.second, nrrd_handle);
    add_bundle(bundle); 
  }

  // ITKDataTypeSignal *return_event = new ITKDataTypeSignal(img);
  //  event = return_event;
  return CONTINUE_E;
#endif
}



BaseTool::propagation_state_e 
Painter::ITKImageFileWrite(event_handle_t event) {
#ifndef HAVE_INSIGHT
  return NrrdFileWrite(event);
#else
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);

  const string &filename = signal->get_vars()->get_string("filename");

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





BaseTool::propagation_state_e 
Painter::ITKBinaryDilateErode(event_handle_t event) {
#ifdef HAVE_INSIGHT

  string name = "ITKBinaryDilateErode";
  typedef itk::BinaryBallStructuringElement< float, 3> StructuringElementType;
  typedef itk::BinaryDilateImageFilter
    < Painter::ITKImageFloat3D, 
    Painter::ITKImageFloat3D,
    StructuringElementType > FilterType;
  FilterType::Pointer filter = FilterType::New();

  StructuringElementType structuringElement;
  structuringElement.SetRadius
    (get_vars()->get_int(name+"::radius"));
  structuringElement.CreateStructuringElement();

  filter->SetKernel(structuringElement);
  filter->SetDilateValue
    (get_vars()->get_double(name+"::dilateValue"));

  typedef itk::BinaryErodeImageFilter
    < Painter::ITKImageFloat3D, 
    Painter::ITKImageFloat3D, 
    StructuringElementType > FilterType2;
  FilterType2::Pointer filter2 = FilterType2::New();

  filter2->SetKernel(structuringElement);
  filter2->SetErodeValue
    (get_vars()->get_double(name+"::erodeValue"));

  NrrdVolume *vol = new NrrdVolume(current_volume_, name, 2);
  volume_map_[name] = vol;
  show_volume(name);
  recompute_volume_list();

#if 0
  get_vars()->insert("ToolDialog::text", 
                     " ITK Binary Dilate Filter...",
                     "string", true);
  get_vars()->insert("Painter::progress_bar_text", "", "string", true);
  get_vars()->unset("ProgressBar::bar_height");
  get_vars()->insert("Painter::progress_bar_total_width","500","string", true);
  get_vars()->insert("ToolDialog::button_height", "0", "string", true); 
#endif

  redraw_all();

  do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);

#if 0
  get_vars()->insert("ToolDialog::text", 
                     " ITK Binary Erode Filter...",
                     "string", true);
  get_vars()->insert("Painter::progress_bar_text", "", "string", true);
  get_vars()->unset("ProgressBar::bar_height");
  get_vars()->insert("Painter::progress_bar_total_width","500","string", true);
  get_vars()->insert("ToolDialog::button_height", "0", "string", true); 
#endif

  redraw_all();

  do_itk_filter<Painter::ITKImageFloat3D>(filter2, vol->nrrd_handle_);

  set_all_slices_tex_dirty();
  redraw_all();
  current_volume_ = vol;  
#endif
  return CONTINUE_E;
}




BaseTool::propagation_state_e 
Painter::ITKGradientMagnitude(event_handle_t) {
#ifdef HAVE_INSIGHT
#if 0
  get_vars()->insert("ToolDialog::text", 
                     " ITK Gradient Magnigude Filter:",
                     "string", true);
  get_vars()->insert("ToolDialog::button_height", "0", "string", true);
  get_vars()->insert("Painter::progress_bar_text", "", "string", true);
  get_vars()->unset("ProgressBar::bar_height");
  get_vars()->insert("Painter::progress_bar_total_width","500","string", true);
#endif
  redraw_all();

  string name = "ITKGradientMagnitude";
  typedef itk::GradientMagnitudeImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
  FilterType::Pointer filter = FilterType::New();

  NrrdVolume *vol = new NrrdVolume(current_volume_, name, 2);
  volume_map_[name] = vol;
  show_volume(name);
  recompute_volume_list();

  do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);
  vol->reset_data_range();
 
  current_volume_ = vol;
  
  set_all_slices_tex_dirty();
  redraw_all();
#endif
  return CONTINUE_E;
}






BaseTool::propagation_state_e 
Painter::ITKCurvatureAnisotropic(event_handle_t event) {
#ifdef HAVE_INSIGHT

#if 0
  get_vars()->insert("ToolDialog::text", 
                     " ITK Curvature Anisotropic Diffusion Filter:",
                     "string", true);
  get_vars()->insert("Painter::progress_bar_text", "", "string", true);
  get_vars()->unset("ProgressBar::bar_height");
  get_vars()->insert("Painter::progress_bar_total_width","500","string", true);
  get_vars()->insert("ToolDialog::button_height", "0", "string", true);
#endif

  redraw_all();

  typedef itk::CurvatureAnisotropicDiffusionImageFilter
    < Painter::ITKImageFloat3D, Painter::ITKImageFloat3D > FilterType;
  FilterType::Pointer filter = FilterType::New();  

  string name = "ITKCurvatureAnisotropic";
  string prefix = "ITKCurvatureAnisotropicDiffusionImageFilterTool::";
  filter->SetNumberOfIterations
    (get_vars()->get_int(prefix+"numberOfIterations"));
  filter->SetTimeStep
    (get_vars()->get_double(prefix+"timeStep"));
  filter->SetConductanceParameter
    (get_vars()->get_double(prefix+"conductanceParameter"));
  
  cerr << "iterations: " << filter->GetNumberOfIterations() << std::endl;

  NrrdVolume *vol = new NrrdVolume(current_volume_, name, 2);
  volume_map_[name] = vol;
  show_volume(name);
  recompute_volume_list();
  
  do_itk_filter<Painter::ITKImageFloat3D>(filter, vol->nrrd_handle_);
  
  current_volume_ = vol;
  set_all_slices_tex_dirty();
  redraw_all();
#endif
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::ITKConfidenceConnected(event_handle_t event) {

#if 0
  get_vars()->insert("ToolDialog::text", 
                     " ITK Confidence Connected Filter: Place Seed Point",
                     "string", true);
  get_vars()->insert("ProgressBar::bar_height","0","string",true);
  get_vars()->unset("ToolDialog::button_height");
  get_vars()->insert("Painter::progress_bar_text", "", "string", true);
#endif

  redraw_all();

#ifdef HAVE_INSIGHT
  tm_.add_tool(new ITKConfidenceConnectedImageFilterTool(this),25); 
#endif
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::ITKThresholdLevelSet(event_handle_t event) {
#ifdef HAVE_INSIGHT

#if 0
  get_vars()->insert
    ("ToolDialog::text", 
     " ITK Threshold Segmentation Level Set Filter: Choose seed layer...",
     "string", true);
  get_vars()->insert("Painter::progress_bar_text", "", "string", true);
  get_vars()->insert("ProgressBar::bar_height","0","string", true);
  get_vars()->insert("Painter::progress_bar_total_width","0","string", true);
  get_vars()->unset("ToolDialog::button_height");
#endif

  redraw_all();

  tm_.add_tool(new BrushTool(this), 25);
  tm_.add_tool(new ITKThresholdTool(this), 26);
#endif
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::FinishTool(event_handle_t event) {
  tm_.propagate_event(new FinishEvent());
  return CONTINUE_E;
}

BaseTool::propagation_state_e 
Painter::CancelTool(event_handle_t event) {
  tm_.propagate_event(new QuitEvent());
  return CONTINUE_E;
}

BaseTool::propagation_state_e 
Painter::SetLayer(event_handle_t event) {
  tm_.propagate_event(new SetLayerEvent());
  return CONTINUE_E;
}

#if 0
BaseTool::propagation_state_e 
Painter::ReloadVolumeTexture(event_handle_t event) {
  if (volume_texture_.get_rep())
    volume_texture_->set_dirty(true);
  return CONTINUE_E;
}
#endif


BaseTool::propagation_state_e 
Painter::ShowVolumeRendering(event_handle_t event)
{
  event_handle_t scene_event = 0;

  if (!current_volume_) return STOP_E;
  NrrdDataHandle nrrd_handle = current_volume_->nrrd_handle_;

  
  NrrdDataHandle volnrrd = new NrrdData();
  NrrdRange *range = nrrdRangeNewSet(nrrd_handle->nrrd_, nrrdBlind8BitRangeState);
  nrrdQuantize(volnrrd->nrrd_, nrrd_handle->nrrd_, range, 8);
  nrrdRangeNix(range);

  const int card_mem = 128;
  volume_texture_ = new Texture;
  NrrdTextureBuilderAlgo::build_static(volume_texture_,
                                       nrrd_handle, 0, 255,
                                       0, 0, 255, card_mem);

    
  const char *colormap = sci_getenv("PAINTER_CMAP2");
  if (!colormap) {
    cerr << "no colormap file specified1\n";
    return STOP_E;
  }

  string fn = string(colormap);
  Piostream *stream = auto_istream(fn, 0);
  if (!stream) {
    cerr << "Error reading file '" + fn + "'." << std::endl;
    return STOP_E;
  }  
  // read the file.
  ColorMap2 *cmap2 = new ColorMap2();
  ColorMap2Handle icmap = cmap2;
  try {
    Pio(*stream, icmap);
  } catch (...) {
    cerr << "Error loading "+fn << std::endl;
    icmap = 0;
    delete stream;
    return STOP_E;
  }
  delete stream;
  ColorMapHandle cmap;
  vector<ColorMap2Handle> *cmap2v = new vector<ColorMap2Handle>(0);
  cmap2v->push_back(icmap);

  vector<Plane *> *planes = new vector<Plane *>;
  VolumeRenderer *vol = new VolumeRenderer(volume_texture_, 
                                           cmap, 
                                           *cmap2v, 
                                           *planes,
                                           Round(card_mem*1024*1024*0.8));
  vol->set_shading(true);
  vol->set_slice_alpha(-0.5);
  vol->set_slice_alpha(0.1);
  vol->set_interactive_rate(4.0);
  vol->set_sampling_rate(3.5);
  vol->set_material(0.322, 0.868, 1.0, 18);
  vol->set_interp(0);
  scene_event = new SceneGraphEvent(vol, "FOO");
  //EventManager::add_event(scene_event);
  process_event(scene_event);
  return CONTINUE_E;
}




BaseTool::propagation_state_e 
Painter::LoadColorMap1D(event_handle_t event) {   
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  const string &fn = signal->get_vars()->get_string("filename");
  Piostream *stream = auto_istream(fn, 0);
  if (!stream) {
    cerr << "Error reading file '" + fn + "'." << std::endl;
    return STOP_E;
  }  

  // read the file.
  ColorMapHandle cmaph = 0;
  try {
    Pio(*stream, cmaph);
  } catch (...) {
    cerr << "Error loading "+fn << std::endl;
    cmaph = 0;
    delete stream;
    return STOP_E;
  }
  delete stream;

  Bundle *bundle = new Bundle();
  bundle->setColorMap(fn, cmaph);
  add_bundle(bundle);
  return CONTINUE_E;
}



BaseTool::propagation_state_e 
Painter::ResampleVolume(event_handle_t event) {   
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);

  NrrdResampleInfo *info = nrrdResampleInfoNew();

  NrrdKernel *kern = 0;;
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0;
  
#if 0
  string last_filtertype_ = "gaussian";

  if (last_filtertype_ == "box") {
    kern = nrrdKernelBox;
  } else if (last_filtertype_ == "tent") {
    kern = nrrdKernelTent;
  } else if (last_filtertype_ == "cubicCR") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 0; 
    p[2] = 0.5; 
  } else if (last_filtertype_ == "cubicBS") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 1; 
    p[2] = 0; 
  } else if (last_filtertype_ == "gaussian") { 
    kern = nrrdKernelGaussian; 
    //    p[0] = sigma_.get(); 
    //    p[1] = extent_.get(); 
  } else  { // default is quartic
#endif
    {
      kern = nrrdKernelAQuartic; 
      p[1] = 0.0834; // most accurate as per Teem documenation
    }

  Nrrd *nin = current_volume_->nrrd_handle_->nrrd_;

  vector<string> samples;
  samples.push_back("1");
  samples.push_back(get_vars()->get_string("Resample::x"));
  samples.push_back(get_vars()->get_string("Resample::y"));
  samples.push_back(get_vars()->get_string("Resample::z"));

  for (int a = 0; a < 4; a++) {
    if (a == 0)
      info->kernel[a] = 0;
    else 
      info->kernel[a] = kern;

    info->samples[a] = nin->axis[a].size;
    int temp;
    bool convert = string_to_int(samples[a], temp);
    ASSERT(convert);
    info->samples[a] = size_t(temp);
           
    memcpy(info->parm[a], p, NRRD_KERNEL_PARMS_NUM * sizeof(double));

    
    if (info->kernel[a] && 
    	(!(airExists(nin->axis[a].min) && airExists(nin->axis[a].max)))) {
      nrrdAxisInfoMinMaxSet(nin, a, nin->axis[a].center ? 
                            nin->axis[a].center : nrrdDefaultCenter);

    
    }
    

    info->min[a] = nin->axis[a].min;
    info->max[a] = nin->axis[a].max;
  }    
  info->boundary = nrrdBoundaryBleed;
  info->type = nin->type;
  info->renormalize = AIR_TRUE;

  NrrdDataHandle nrrd_handle = scinew NrrdData;
  Nrrd *nout = nrrd_handle->nrrd_;
  if (nrrdSpatialResample(nout, nin, info)) {
    char *err = biffGetDone(NRRD);
    string errstr(err);
    free(err);
    throw "Trouble resampling: " + errstr;

  }
  nrrdResampleInfoNix(info); 

  //  current_volume_->nrrd_handle_ = nrrd_handle;

  //  NrrdDataHandle nrrd_handle = scinew NrrdData;
  string newname = current_volume_->name_+" - Resampled";
  NrrdVolume *vol = new NrrdVolume(0, newname, nrrd_handle);
  volume_map_[newname] = vol;
  show_volume(newname);
  recompute_volume_list();

  //  recompute_volume_list();

  return CONTINUE_E;

}
  



BaseTool::propagation_state_e 
Painter::AbortFilterOn(event_handle_t event) {
  abort_filter_ = true;
  return CONTINUE_E;
}

  
} // end namespace SCIRun
