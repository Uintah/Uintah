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
//    File   : PainterSignalTargets.cc
//    Author : McKay Davis
//    Date   : Mon Jul  3 21:47:22 2006


#include <StandAlone/Apps/Seg3D/Painter.h>
#include <StandAlone/Apps/Seg3D/CropTool.h>
#include <StandAlone/Apps/Seg3D/BrushTool.h>
#include <StandAlone/Apps/Seg3D/ITKThresholdSegmentationLevelSetImageFilterTool.h>
#include <StandAlone/Apps/Seg3D/ITKConfidenceConnectedImageFilterTool.h>
#include <StandAlone/Apps/Seg3D/FloodfillTool.h>
#include <StandAlone/Apps/Seg3D/SessionReader.h>
#include <StandAlone/Apps/Seg3D/SessionWriter.h>
#include <StandAlone/Apps/Seg3D/VolumeOps.h>
#include <StandAlone/Apps/Seg3D/VolumeFilter.h>

#include <sci_comp_warn_fixes.h>
#include <iostream>
#include <sci_gl.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomTransform.h>
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
#include <Core/Skinner/GeomSkinnerVarSwitch.h>

#ifndef _WIN32
#  include <sys/mman.h>
#else
#  include <Core/OS/Rand.h>
#  include <io.h>
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

  REGISTER_CATCHER_TARGET(Painter::MemMapFileRead);

  REGISTER_CATCHER_TARGET(Painter::CancelTool);  
  REGISTER_CATCHER_TARGET(Painter::FinishTool);
  REGISTER_CATCHER_TARGET(Painter::SetLayer);
  REGISTER_CATCHER_TARGET(Painter::LoadColorMap1D);

  REGISTER_CATCHER_TARGET(Painter::ITKBinaryDilate);  
  REGISTER_CATCHER_TARGET(Painter::ITKGradientMagnitude);
  REGISTER_CATCHER_TARGET(Painter::ITKBinaryDilateErode);
  REGISTER_CATCHER_TARGET(Painter::ITKCurvatureAnisotropic);
  REGISTER_CATCHER_TARGET(Painter::start_ITKConfidenceConnectedImageFilterTool);
  REGISTER_CATCHER_TARGET(Painter::start_ITKThresholdSegmentationLevelSetImageFilterTool);

  REGISTER_CATCHER_TARGET(Painter::ShowVolumeRendering);
  REGISTER_CATCHER_TARGET(Painter::ShowIsosurface);

  REGISTER_CATCHER_TARGET(Painter::AbortFilterOn);

  REGISTER_CATCHER_TARGET(Painter::LoadSession);
  REGISTER_CATCHER_TARGET(Painter::SaveSession);

  REGISTER_CATCHER_TARGET(Painter::LoadVolume);
  REGISTER_CATCHER_TARGET(Painter::SaveVolume);

  REGISTER_CATCHER_TARGET(Painter::ResampleVolume);

  REGISTER_CATCHER_TARGET(Painter::CreateLabelVolume);  
  REGISTER_CATCHER_TARGET(Painter::CreateLabelChild);

  REGISTER_CATCHER_TARGET(Painter::RebuildLayers);
   
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
  if (tm_.query_tool_id(25) != "") {
    return STOP_E;
  }
  tm_.add_tool(new BrushTool(this),25); 
  return CONTINUE_E;
}
  

BaseTool::propagation_state_e 
Painter::StartCropTool(event_handle_t event) {
  if (tm_.query_tool_id(25) != "") {
    return STOP_E;
  }
  tm_.add_tool(new CropTool(this),25);
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::StartFloodFillTool(event_handle_t event) {
  tm_.add_tool(new FloodfillTool(this),25);
  return CONTINUE_E;
}




BaseTool::propagation_state_e 
Painter::Autoview(event_handle_t) {
  if (current_volume_.get_rep()) {
    SliceWindows::iterator window = windows_.begin();
    SliceWindows::iterator end = windows_.end();
    for (;window != end; ++window) {
      (*window)->autoview(current_volume_);
    }
  }
  AutoviewEvent *autoview_event = new AutoviewEvent();
  EventManager::add_event(autoview_event);
  return CONTINUE_E;
}




BaseTool::propagation_state_e 
Painter::CopyLayer(event_handle_t) {
  if (!current_volume_.get_rep()) return STOP_E;
  NrrdDataHandle &nrrdh = current_volume_->nrrd_handle_;
  nrrdh.detach();
  string name = unique_layer_name(current_volume_->name_);
  volumes_.push_back(new NrrdVolume(this, name, nrrdh, current_volume_->label_));
  rebuild_layer_buttons();
  redraw_all();
  return CONTINUE_E;
}

BaseTool::propagation_state_e 
Painter::DeleteLayer(event_handle_t event) {
  NrrdVolumeHandle &layer = current_volume_;
  if (event.get_rep()) {
    Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
    ASSERT(signal);
    Skinner::Variables *vars = signal->get_vars();
    layer = find_volume_by_name(vars->get_string("LayerButton::name"));
  }

  if (!layer.get_rep()) return STOP_E;

  NrrdVolumeHandle parent = layer->parent_;
  NrrdVolumes &volumes =  parent.get_rep() ? parent->children_ : volumes_;

  NrrdVolumes newvolumes(volumes.size()-1, 0);
  int j = 0;
  unsigned int newcur = 0;
  for (unsigned int i = 0; i < volumes.size(); ++i) {
    if (volumes_[i] != layer)
      newvolumes[j++] = volumes[i];
    else 
      newcur = i;
  }
  volumes = newvolumes;
  layer = 0;
  if (newcur < volumes.size()) {
    current_volume_ = volumes[newcur];
  } else if (parent.get_rep()) {
    current_volume_ = parent;
  } else {
    current_volume_ = 0;
  }
  rebuild_layer_buttons();
  extract_all_window_slices();
  redraw_all();

  return CONTINUE_E;
}

BaseTool::propagation_state_e 
Painter::NewLayer(event_handle_t event) {
  if (!current_volume_.get_rep()) return STOP_E;
  
  if (current_volume_->label_) {
    return CreateLabelChild(event);
  }
#ifdef HAVE_INSIGHT
  NrrdDataHandle nrrdh = 
    VolumeOps::create_clear_nrrd(current_volume_->nrrd_handle_);
#else
  NrrdDataHandle nrrdh = 0;
#endif
  string name = unique_layer_name(current_volume_->name_);
  volumes_.push_back(new NrrdVolume(this, name, nrrdh));
  rebuild_layer_buttons();
  redraw_all();
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
  volumes_.push_back(new NrrdVolume(this, filename, nrrd_handle));
  status_ =  "Successfully Loaded Nrrd: "+filename;

  return CONTINUE_E;  
}



BaseTool::propagation_state_e 
Painter::ITKBinaryDilate(event_handle_t event) {
#ifdef HAVE_INSIGHT
  string name = "ITKBinaryDilate";
  typedef itk::BinaryBallStructuringElement< float, 3> StructuringElementType;
  typedef itk::BinaryDilateImageFilter
    < ITKImageFloat3D, ITKImageFloat3D, StructuringElementType > FilterType;
  FilterType::Pointer filter = FilterType::New();

  StructuringElementType structuringElement;
  structuringElement.SetRadius(get_vars()->get_int(name+"::radius"));
  structuringElement.CreateStructuringElement();
  
  filter->SetKernel(structuringElement);
  filter->SetDilateValue(get_vars()->get_double(name+"::dilateValue"));

  NrrdVolumeHandle &vol = current_volume_;
  vol->nrrd_handle_ = 
    do_itk_filter<FilterType>(filter, vol->nrrd_handle_);
  redraw_all();
#endif
  return CONTINUE_E;
}



BaseTool::propagation_state_e 
Painter::LoadVolume(event_handle_t event) {
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  string filename = signal->get_vars()->get_string("filename");
  NrrdVolumeHandle volume = load_volume<float>(filename);
  if (!volume.get_rep()) {
    return STOP_E;
  }

  volumes_.push_back(volume);
  current_volume_ = volume;

  for (unsigned int i = 0; i < windows_.size(); ++ i) {
    windows_[i]->center_ = volume->center();
  }

  extract_all_window_slices();
  rebuild_layer_buttons();
  redraw_all();
  status_ =  "Successfully Loaded File: "+filename;

  return CONTINUE_E;
}



BaseTool::propagation_state_e 
Painter::ITKBinaryDilateErode(event_handle_t event) {
#ifndef HAVE_INSIGHT
  cerr << "Insight not compiled\n";
  return STOP_E;
#else
  string name = "ITKBinaryDilateErode";
  typedef itk::BinaryBallStructuringElement< float, 3> StructuringElementType;
  typedef itk::BinaryDilateImageFilter
    < ITKImageFloat3D, 
    ITKImageFloat3D,
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
    < ITKImageFloat3D, 
    ITKImageFloat3D, 
    StructuringElementType > FilterType2;
  FilterType2::Pointer filter2 = FilterType2::New();

  filter2->SetKernel(structuringElement);
  filter2->SetErodeValue
    (get_vars()->get_double(name+"::erodeValue"));

  NrrdVolumeHandle vol = 
    new NrrdVolume(this, name, current_volume_->nrrd_handle_);
  volumes_.push_back(vol);

  vol->nrrd_handle_ = do_itk_filter<FilterType>(filter, vol->nrrd_handle_);
  vol->nrrd_handle_ = do_itk_filter<FilterType2>(filter2, vol->nrrd_handle_);

  set_all_slices_tex_dirty();
  redraw_all();
  current_volume_ = vol;
  return CONTINUE_E;
#endif
}




BaseTool::propagation_state_e 
Painter::ITKGradientMagnitude(event_handle_t) {
#ifndef HAVE_INSIGHT
  cerr << "Insight not compiled\n";
  return STOP_E;
#else
  typedef itk::GradientMagnitudeImageFilter
    < ITKImageFloat3D, ITKImageFloat3D > FilterType;

  VolumeFilter<FilterType>(copy_current_layer(" Gradient Magnitude"))();
  current_volume_->reset_data_range();
  extract_all_window_slices();
  redraw_all();
#endif
  return CONTINUE_E;
}






BaseTool::propagation_state_e 
Painter::ITKCurvatureAnisotropic(event_handle_t event) {
#ifndef HAVE_INSIGHT
  cerr << "Insight not compiled\n";
  return STOP_E;
#else
  typedef itk::CurvatureAnisotropicDiffusionImageFilter
    < ITKImageFloat3D, ITKImageFloat3D > FilterType;
  FilterType::Pointer filter = FilterType::New();  

  string name = "ITKCurvatureAnisotropic";
  string prefix = "ITKCurvatureAnisotropicDiffusionImageFilterTool::";
  filter->SetNumberOfIterations
    (get_vars()->get_int(prefix+"numberOfIterations"));
  filter->SetTimeStep
    (get_vars()->get_double(prefix+"timeStep"));
  filter->SetConductanceParameter
    (get_vars()->get_double(prefix+"conductanceParameter"));

  name = unique_layer_name(name);
  NrrdDataHandle nrrdh = current_volume_->nrrd_handle_;
  nrrdh.detach();
  NrrdVolume *vol = new NrrdVolume(this, name, nrrdh);
  volumes_.push_back(vol);
  current_volume_ = vol;
  filter_volume_ = vol;
  rebuild_layer_buttons();
  
  vol->nrrd_handle_ = do_itk_filter<FilterType>(filter, current_volume_->nrrd_handle_);
  
  vol->dirty_ = true;
  extract_all_window_slices();
  redraw_all();
  return CONTINUE_E;
#endif
}


BaseTool::propagation_state_e 
Painter::start_ITKConfidenceConnectedImageFilterTool(event_handle_t event) {
#ifndef HAVE_INSIGHT
  cerr << "Insight not compiled\n";
  return STOP_E;
#else
  tm_.add_tool(new ITKConfidenceConnectedImageFilterTool(this),25); 
  return CONTINUE_E;
#endif
  
}


BaseTool::propagation_state_e 
Painter::start_ITKThresholdSegmentationLevelSetImageFilterTool
(event_handle_t event) {
#ifndef HAVE_INSIGHT
   cerr << "Insight not compiled\n";
  return STOP_E;
#else
  tm_.add_tool(new BrushTool(this), 25);
  tm_.add_tool(new ITKThresholdSegmentationLevelSetImageFilterTool(this), 26);
  return CONTINUE_E;
#endif
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

#if defined(_WIN32) && !defined(BUILD_CORE_STATIC)
#undef SCISHARE
#define SCISHARE __declspec(dllimport)
#else
#define SCISHARE
#endif


extern SCISHARE GeomHandle fast_lat_mc(Nrrd *nrrd, 
                                       double isoval, 
                                       unsigned int mask);


void
Painter::isosurface_label_volumes(NrrdVolumes &volumes, GeomGroup *group)
{
  for (unsigned int i = 0; i < volumes.size(); ++i) {

    NrrdVolumeHandle volume = volumes[i];
    if (volume->label_) {

      GeomHandle isosurface;
      NrrdDataHandle nrrdh = volume->nrrd_handle_;
      if (volume->label_ && nrrdh->nrrd_->type == nrrdTypeUInt) 
      {

        isosurface = fast_lat_mc(nrrdh->nrrd_, 
                                 volume->label_/2.0,
                                 volume->label_);
      } else {
        isosurface = fast_lat_mc(volume->nrrd_handle_->nrrd_, 0.0,0);
      }

      GeomMaterial *colored_isosurface =
        new GeomMaterial(isosurface, 
                         volume->get_colormap()->lookup2(volume->bit()+1));
      GeomSkinnerVarSwitch *volume_isosurface = 
        new GeomSkinnerVarSwitch(colored_isosurface, volume->visible_);

      group->add(volume_isosurface);
    }
    if (!volume->children_.empty())
      isosurface_label_volumes(volume->children_, group);
  }
}




BaseTool::propagation_state_e 
Painter::ShowIsosurface(event_handle_t event)
{
  //static int count = 0;
  if (!current_volume_.get_rep()) return STOP_E;
  //  event_handle_t scene_event = = ;

  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
                
  Matrix &tmat = current_volume_->transform_;
  Transform transform(current_volume_->min(), 
                      Vector(tmat.get(1,1), 0, 0),
                      Vector(0, tmat.get(2,2), 0),
                      Vector(0, 0, tmat.get(3,3)));
  
  GeomGroup *group = new GeomGroup();
  isosurface_label_volumes(volumes_, group);
  GeomTransform *everything = new GeomTransform(group, transform);
  EventManager::add_event(new SceneGraphEvent(everything, "Transparent IsoSurface"));

  return CONTINUE_E;
}
  

BaseTool::propagation_state_e 
Painter::ShowVolumeRendering(event_handle_t event)
{
  event_handle_t scene_event = 0;

  if (!current_volume_.get_rep()) return STOP_E;

  const int card_mem = 128;
  volume_texture_ = new Texture;
  NrrdTextureBuilderAlgo::build_static(volume_texture_,
                                       current_volume_->nrrd_handle_, 0, 255,
                                       0, 0, 255, card_mem);

  Skinner::Var<string> filename(get_vars(), "cmap2_filename", "default.cmap2");
    
  string path = findFileInPath(filename(),sci_getenv("SKINNER_PATH"));
  if (path.empty()) {
    cerr << "no colormap file specified1\n";
    return STOP_E;
  }

  string fn = path + filename();

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

  icmap->value_range().first = 0;
  icmap->value_range().second = 1300;

  vector<Plane *> *planes = new vector<Plane *>;
  VolumeRenderer *vol = new VolumeRenderer(volume_texture_, 
                                           cmap, 
                                           *cmap2v, 
                                           *planes,
                                           Round(card_mem*1024*1024*0.8));
  vol->set_mode(TextureRenderer::MODE_OVER);
  vol->set_shading(true);
  vol->set_sw_raster(true);
  //  vol->set_slice_alpha(-0.5);
  //vol->set_slice_alpha(1.1);
  vol->set_interactive_rate(1.0);
  vol->set_sampling_rate(1.5);
  vol->set_material(0.322, 0.868, 1.0, 18);
  //vol->set_interp(0);
  scene_event = new SceneGraphEvent(vol, "FOO");
  EventManager::add_event(scene_event);
  //process_event(scene_event);
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

  colormaps_.push_back(cmaph);
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
  volumes_.push_back(vol);
  return CONTINUE_E;

}
  



BaseTool::propagation_state_e 
Painter::AbortFilterOn(event_handle_t event) {
  cerr << "Stop Fitler\n";
#ifdef HAVE_INSIGHT
  for (unsigned int i = 0; i < filters_.size(); ++i) 
    filters_[i]->stop();
  //  abort_filter_ = true;
#endif
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::CreateLabelVolume(event_handle_t event) {
  if (!current_volume_.get_rep()) return STOP_E;
  volume_lock_.lock();
#ifdef HAVE_INSIGHT
  NrrdDataHandle nrrdh = 
    VolumeOps::create_clear_nrrd(current_volume_->nrrd_handle_, nrrdTypeUInt);
#else
  NrrdDataHandle nrrdh = 0;
#endif
  string name = unique_layer_name(current_volume_->name_ + " Label");
  volumes_.push_back(new NrrdVolume(this, name, nrrdh, 1));
  current_volume_ = volumes_.back();
  volume_lock_.unlock();
  extract_all_window_slices();
  rebuild_layer_buttons();
  redraw_all();
  return CONTINUE_E;
}


BaseTool::propagation_state_e 
Painter::CreateLabelChild(event_handle_t event) {
  if (!current_volume_.get_rep()) return STOP_E;
  current_volume_ = current_volume_->create_child_label_volume();
  //  volumes_.push_back();
  //  current_volume_ = volumes_.back();
  extract_all_window_slices();
  rebuild_layer_buttons();
  redraw_all();
  return CONTINUE_E;
}

BaseTool::propagation_state_e 
Painter::LoadSession(event_handle_t event) {
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  string filename = signal->get_vars()->get_string("filename");
  
  SessionReader reader(this);
  if (reader.load_session(filename)) {
    if (current_volume_.get_rep()) 
      for (unsigned int i = 0; i < windows_.size(); ++ i) {
        windows_[i]->center_ = current_volume_->center();
      }

    status_ =  "Successfully loaded sesion: "+filename;
  } else {
    status_ =  "Error loading session "+filename;
  }

  return CONTINUE_E;
}
  


BaseTool::propagation_state_e 
Painter::SaveSession(event_handle_t event) {
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  string filename = signal->get_vars()->get_string("filename");
  
  if ( SessionWriter::write_session(filename, volumes_)) {
    status_ =  "Successfully saved sesion: "+filename;
  } else {
    status_ =  "Error loading session "+filename;
  }

  return CONTINUE_E;
}
  


BaseTool::propagation_state_e 
Painter::SaveVolume(event_handle_t event) {
  if (!current_volume_.get_rep()) {
    status_ = "No Layer Selected";
    return STOP_E;
  }

  NrrdVolumeHandle parent = current_volume_;
  while (parent->parent_.get_rep()) parent = parent->parent_;
  Skinner::Signal *signal = dynamic_cast<Skinner::Signal *>(event.get_rep());
  ASSERT(signal);
  string filename = signal->get_vars()->get_string("filename");
  if (parent->write(filename)) {
    status_ = "Successfully saved layer: " + filename;
  } else {
    status_ = "Error saving layer: " + filename;
  }
  
  return CONTINUE_E;
}




BaseTool::propagation_state_e 
Painter::RebuildLayers(event_handle_t) {
  rebuild_layer_buttons();
  return CONTINUE_E;
}

  
} // end namespace SCIRun
