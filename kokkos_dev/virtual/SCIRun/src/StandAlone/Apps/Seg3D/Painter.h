/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  Painter.h
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */


#ifndef StandAlone_Apps_Painter_Painter_h
#define StandAlone_Apps_Painter_Painter_h


#include <sci_comp_warn_fixes.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>

#include <StandAlone/Apps/Seg3D/VolumeSlice.h>
#include <StandAlone/Apps/Seg3D/NrrdVolume.h>
#include <StandAlone/Apps/Seg3D/SliceWindow.h>
#include <StandAlone/Apps/Seg3D/LayerButton.h>
#include <StandAlone/Apps/Seg3D/UIvar.h>
#include <StandAlone/Apps/Seg3D/NrrdToITK.h>

#include <Core/Algorithms/Visualization/RenderField.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/TextRenderer.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geom/GeomCull.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/IndexedGroup.h>
#include <Core/Geom/TexSquare.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Geom/FreeType.h>
#include <Core/Geometry/Plane.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Timer.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FileUtils.h>
#include <Core/Volume/CM2Widget.h>
#include <Core/Skinner/Parent.h>
#include <Core/Skinner/Color.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/Tools/BaseTool.h>
#include <Core/Events/Tools/ToolManager.h>
#include <Core/Volume/Texture.h>

#include <sci_defs/insight_defs.h>

#ifdef HAVE_INSIGHT
#  include <Core/Datatypes/ITKDatatype.h>
#  include <itkImageToImageFilter.h>
#  include <itkImageFileReader.h>
#  include <itkImportImageFilter.h>
#  include <itkImageIOBase.h>
#  include <itkCommand.h>
#  include <itkThresholdSegmentationLevelSetImageFilter.h>
#endif

#ifdef _WIN32
#undef min
#undef max
#endif

namespace SCIRun {

#ifdef HAVE_INSIGHT
using SCIRun::ITKDatatypeHandle;
typedef itk::Image<float,3> ITKImageFloat3D;
#endif


class VolumeFilterBase;

class Painter : public Skinner::Parent
{
public:
  Painter(Skinner::Variables *, VarContext* ctx);
  virtual ~Painter();

  static Skinner::DrawableMakerFunc_t   maker;
  static string                         class_name() { return "Painter"; }
  void                                  redraw_all();
  void                                  extract_all_window_slices();
private:
  friend class SliceWindow;
  friend class VolumeSlice;
  friend class NrrdVolume;
  friend class LayerButton; 
  friend class PointerToolSelectorTool;
  friend class KeyToolSelectorTool;
  friend class CLUTLevelsTool;
  friend class ZoomTool;
  friend class PainterAutoviewTool;
  friend class ProbeTool;
  friend class PanTool;
  friend class LayerMergeTool;
  friend class CropTool;
  friend class BrushTool;
  friend class FloodfillTool;
  friend class ITKThresholdSegmentationLevelSetImageFilterTool;
  friend class StatisticsTool;
  friend class ITKConfidenceConnectedImageFilterTool;
  friend class SessionReader;

  enum DisplayMode_e {
    normal_e,
    slab_e,
    mip_e,
    num_display_modes_e
  };

  // Methods for drawing to the GL window

  void                  set_probe();

  // these should probably be moved to NrrdVolume class
  template <class T>
  NrrdVolumeHandle      load_volume(string);
  bool                  save_volume(string filename, NrrdVolumeHandle &);
  
  void                  set_all_slices_tex_dirty();
  ColorMapHandle        get_colormap(int);
  void                  rebuild_layer_buttons();
  void                  get_data_from_layer_buttons();

  void                  build_layer_button(unsigned int &, NrrdVolumeHandle &);
  void                  build_volume_list(NrrdVolumes &,
                                          NrrdVolumeHandle &vol);

  void                  move_layer_up(NrrdVolumeHandle &);
  void                  move_layer_down(NrrdVolumeHandle &);
  bool                  merge_layer(NrrdVolumeHandle &);
  void                  cur_layer_up();
  void                  cur_layer_down();
  void                  opacity_up();
  void                  opacity_down();
  void                  reset_clut();
  void                  create_undo_volume();
  void                  undo_volume();
  NrrdVolumeHandle      find_volume_by_name(const string &);

  void                  isosurface_label_volumes(NrrdVolumes &, GeomGroup *);
  string		unique_layer_name(string);

  NrrdVolumeHandle      copy_current_layer(string suffix = "");
  NrrdVolumeHandle      make_layer(string suffix,
                                   NrrdDataHandle &,
                                   unsigned int mask=0);


#ifdef HAVE_INSIGHT
  template <class FilterType>
  NrrdDataHandle        do_itk_filter (FilterType *, NrrdDataHandle &);

  template <class FilterType>
  void                  filter_callback(itk::Object *, 
                                        const itk::EventObject &);

  vector<VolumeFilterBase *> filters_;

  //  void                  filter_callback_const (const itk::Object *, 
  //                                               const itk::EventObject &);
#endif
  
  CatcherFunction_t     InitializeSignalCatcherTargets;
  CatcherFunction_t     SliceWindow_Maker;
  CatcherFunction_t     LayerButton_Maker;

  CatcherFunction_t     StartBrushTool;
  CatcherFunction_t     StartCropTool;
  CatcherFunction_t     StartFloodFillTool;

  CatcherFunction_t     Autoview;
  CatcherFunction_t     CopyLayer;
  CatcherFunction_t     DeleteLayer;
  CatcherFunction_t     NewLayer;

  CatcherFunction_t     MemMapFileRead;
  CatcherFunction_t     NrrdFileWrite;

  CatcherFunction_t     FinishTool;
  CatcherFunction_t     CancelTool;
  CatcherFunction_t     SetLayer;
  CatcherFunction_t     LoadColorMap1D;

  CatcherFunction_t     ITKBinaryDilate;
  CatcherFunction_t     ITKBinaryErode;

  CatcherFunction_t     ITKImageFileWrite;
  CatcherFunction_t     ITKGradientMagnitude;
  CatcherFunction_t     ITKBinaryDilateErode;
  CatcherFunction_t     ITKCurvatureAnisotropic;
  CatcherFunction_t     start_ITKConfidenceConnectedImageFilterTool;
  CatcherFunction_t     start_ITKThresholdSegmentationLevelSetImageFilterTool;

  CatcherFunction_t     ShowVolumeRendering;
  CatcherFunction_t     ShowIsosurface;
  CatcherFunction_t     AbortFilterOn;

  CatcherFunction_t     LoadVolume;
  CatcherFunction_t     SaveVolume;

  CatcherFunction_t     ResampleVolume;
  CatcherFunction_t     CreateLabelVolume;
  CatcherFunction_t     CreateLabelChild;

  CatcherFunction_t     LoadSession;
  CatcherFunction_t     SaveSession;

  CatcherFunction_t     RebuildLayers;


  typedef vector<ColorMapHandle> ColorMaps_t;
  SliceWindow *         cur_window_;
  ToolManager           tm_;
  Point                 pointer_pos_;
  SliceWindows		windows_;
  NrrdVolumes		volumes_;
  NrrdVolumeHandle      current_volume_;
  ColorMaps_t           colormaps_;
  Mutex                 volume_lock_;

  TextureHandle         volume_texture_;
  vector<LayerButton *> layer_buttons_;
  NrrdVolumeHandle      filter_volume_;
  bool                  abort_filter_;
  Skinner::Var<string>  status_;
  //  Skinner::Drawables_t  filters_;
};




class RedrawSliceWindowEvent : public RedrawEvent 
{
  SliceWindow &        window_;
public:
  RedrawSliceWindowEvent(SliceWindow &window) :
    RedrawEvent(),
    window_(window)
  {
  }
  
  ~RedrawSliceWindowEvent() {}
  SliceWindow &       get_window() { return window_; }
};



class FinishEvent : public QuitEvent 
{
public:
  FinishEvent() : QuitEvent() {}
  ~FinishEvent() {}
};

class SetLayerEvent : public RedrawEvent 
{
public:
  SetLayerEvent() : RedrawEvent() {}
  ~SetLayerEvent() {}
};




template <class T>
unsigned int max_vector_magnitude_index(vector<T> array) {
  if (array.empty()) return 0;
  unsigned int index = 0;
  for (unsigned int i = 1; i < array.size(); ++i) 
    if (fabs(array[i]) > fabs(array[index]))
      index = i;
  return index;
}


#ifdef HAVE_INSIGHT

template <class FilterType>
NrrdDataHandle
Painter::do_itk_filter(FilterType *filter,
                       NrrdDataHandle &nrrd_handle) 
{
  typedef typename FilterType::InputImageType ImageInT;
  typedef typename FilterType::OutputImageType ImageOutT;

  typedef typename itk::MemberCommand< Painter > RedrawCommandType;
  typename RedrawCommandType::Pointer callback = RedrawCommandType::New();
  callback->SetCallbackFunction
    (this, &Painter::filter_callback<FilterType>);
  //  callback->SetCallbackFunction(this, &Painter::filter_callback_const);
  filter->AddObserver(itk::ProgressEvent(), callback);
  filter->AddObserver(itk::IterationEvent(), callback);
  
  if (nrrd_handle.get_rep()) {
    ITKDatatypeHandle img_handle = nrrd_to_itk_image(nrrd_handle);  
    ImageInT *imgp = dynamic_cast<ImageInT*>(img_handle->data_.GetPointer());
    if (imgp == 0) {
      return 0;
    }    
    filter->SetInput(imgp);
  }

  try {
    filter->Update();
  } catch (itk::ExceptionObject &err) {
    if (!abort_filter_) {
      cerr << "ITK Exception: \n";
      err.Print(cerr);
      return 0;
    } else {
      abort_filter_ = false;
    }
  } catch (...) {
    cerr << "ITK Filter error!\n";
    return 0;
  }
  
  SCIRun::ITKDatatypeHandle output_img = new SCIRun::ITKDatatype();
  output_img->data_ = filter->GetOutput();

  return itk_image_to_nrrd<typename ImageOutT::PixelType>(output_img);
}


template <class FilterType>
void
Painter::filter_callback(itk::Object *object,
                         const itk::EventObject &event)
{
  typedef typename FilterType::InputImageType ImageInT;
  typedef typename FilterType::OutputImageType ImageOutT;

  itk::ProcessObject::Pointer process = 
    dynamic_cast<itk::ProcessObject *>(object);
  ASSERT(process);
  double value = process->GetProgress();
  if (typeid(itk::ProgressEvent) == typeid(event))
  {
    std::cerr << "Filter progress: " << value * 100.0 << "%\n";
    if (filter_volume_.get_rep()) {
      FilterType *filter = dynamic_cast<FilterType *>(object);
      ASSERT(filter);
      volume_lock_.lock();
      ITKDatatypeHandle imgh = new ITKDatatype();
      imgh->data_ = filter->GetOutput();
      ImageOutT *img = dynamic_cast<ImageOutT *>(imgh->data_.GetPointer());
      if (!img) return;
      typedef typename FilterType::OutputImageType::PixelType OutT;
      filter_volume_->nrrd_handle_ = itk_image_to_nrrd<OutT>(imgh);
      filter_volume_->set_dirty();
      volume_lock_.unlock();
      extract_all_window_slices();
      
      //      if (volume_texture_.get_rep()) {
      //  NrrdTextureBuilderAlgo::build_static
      //	  (volume_texture_,current_volume_->nrrd_handle_, 0, 255,
      //	   0, 0, 255, 128);
      //    }
      //      extract_all_window_slices();
      //set_all_slices_tex_dirty();
    }

    redraw_all();

  }


  if (typeid(itk::IterationEvent) == typeid(event))
  {
    std::cerr << "Filter Iteration: " << value * 100.0 << "%\n";
  }

  if (abort_filter_) {
    process->AbortGenerateDataOn();
  }
}

#endif

template <class T>
NrrdVolumeHandle
Painter::load_volume(string filename) {
  filename = substituteTilde(filename);
  if (!validFile(filename)) {
    return 0;
  }

#ifndef HAVE_INSIGHT
  NrrdDataHandle nrrd_handle = new NrrdData();
  if (nrrdLoad(nrrd_handle->nrrd_, filename.c_str(), 0)) {
    return 0;    
  } 
#else
  // create a new reader
  typedef itk::ImageFileReader <itk::Image<T, 3> > FileReaderType;
  typename FileReaderType::Pointer reader = FileReaderType::New();
  
  reader->SetFileName(filename.c_str());

  try {
    reader->Update();  
  } catch  ( itk::ExceptionObject & err ) {
    cerr << "Painter::read_volume - ITK ExceptionObject caught!" << std::endl;
    cerr << err.GetDescription() << std::endl;
    return 0;
  }
  
  SCIRun::ITKDatatype *img = new SCIRun::ITKDatatype();
  img->data_ = reader->GetOutput();

  if (!img->data_) { 
    cerr << "no itk image\n";
    return 0;
  }

  ITKDatatypeHandle img_handle = img;
  NrrdDataHandle nrrd_handle = itk_image_to_nrrd<T>(img_handle);
#endif

  if (!nrrd_handle->nrrd_) return 0;

  pair<string, string> dir_file = split_filename(filename);
  NrrdVolume *vol = new NrrdVolume(this, dir_file.second, nrrd_handle);
  vol->filename_ = dir_file.second;
  return vol;
}


}
#endif
