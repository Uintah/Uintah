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

#include <StandAlone/Apps/Painter/VolumeSlice.h>
#include <StandAlone/Apps/Painter/NrrdVolume.h>
#include <StandAlone/Apps/Painter/SliceWindow.h>
#include <StandAlone/Apps/Painter/LayerButton.h>
#include <StandAlone/Apps/Painter/UIvar.h>

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



class Painter : public Skinner::Parent
{
public:
  Painter(Skinner::Variables *, VarContext* ctx);
  virtual ~Painter();

  static Skinner::DrawableMakerFunc_t   maker;
  static string                         class_name() { return "Painter"; }
  virtual int                           get_signal_id(const string &signalname) const;


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
  friend class ITKThresholdTool;
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
  void			redraw_all();
  void			draw_slice_lines(SliceWindow &);  
  void			extract_all_window_slices();
  void                  set_probe();

  NrrdVolume *          load_volume(const string &);            
  void                  copy_current_layer();
  void                  new_current_layer();
  void                  set_all_slices_tex_dirty();
  ColorMapHandle        get_colormap(int);
  void                  rebuild_layer_buttons();
  void                  get_data_from_layer_buttons();

  void                  build_layer_button(unsigned int &, NrrdVolume *);
  void                  build_volume_list(NrrdVolumes &,NrrdVolume *vol=0);

  void                  move_layer_up(NrrdVolume *);
  void                  move_layer_down(NrrdVolume *);
  void                  cur_layer_up();
  void                  cur_layer_down();
  void                  opacity_up();
  void                  opacity_down();
  void                  reset_clut();
  void                  create_undo_volume();
  void                  undo_volume();
  NrrdVolume *          find_volume_by_name(const string &);
  pair<double, double>  compute_mean_and_deviation(Nrrd *, Nrrd *);
  NrrdVolume *          copy_current_volume(const string &, int mode=0);

  void                  isosurface_label_volumes(NrrdVolumes &, GeomGroup *);

#ifdef HAVE_INSIGHT
  ITKDatatypeHandle     nrrd_to_itk_image(NrrdDataHandle &nrrd);
  NrrdDataHandle        itk_image_to_nrrd(ITKDatatypeHandle &);
  template <class ImageT>
  bool                  do_itk_filter(itk::ImageToImageFilter<ImageT,ImageT> *,
                                      NrrdDataHandle &nrrd);
  void                  filter_callback(itk::Object *, 
                                        const itk::EventObject &);
  void                  filter_callback_const (const itk::Object *, 
                                               const itk::EventObject &);
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
  CatcherFunction_t     MergeLayer;

  CatcherFunction_t     MemMapFileRead;
  CatcherFunction_t     NrrdFileWrite;

  CatcherFunction_t     FinishTool;
  CatcherFunction_t     CancelTool;
  CatcherFunction_t     SetLayer;
  CatcherFunction_t     LoadColorMap1D;

  CatcherFunction_t     ITKBinaryDilate;

  CatcherFunction_t     ITKImageFileWrite;
  CatcherFunction_t     ITKGradientMagnitude;
  CatcherFunction_t     ITKBinaryDilateErode;
  CatcherFunction_t     ITKCurvatureAnisotropic;
  CatcherFunction_t     ITKConfidenceConnected;
  CatcherFunction_t     ITKThresholdLevelSet;

  CatcherFunction_t     ShowVolumeRendering;
  CatcherFunction_t     ShowIsosurface;
  CatcherFunction_t     AbortFilterOn;

  CatcherFunction_t     LoadVolume;
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
  NrrdVolume *          current_volume_;
  NrrdVolume *          undo_volume_;
  ColorMaps_t           colormaps_;
  UIint			anatomical_coordinates_;
  Mutex                 volume_lock_;

  TextureHandle         volume_texture_;
  vector<LayerButton *> layer_buttons_;
  NrrdVolume *          filter_volume_;
  bool                  abort_filter_;
  Skinner::Var<string>  status_;

#ifdef HAVE_INSIGHT
  ITKDatatypeHandle     filter_update_img_;
#endif


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



int nrrd_type_size(Nrrd *);
int nrrd_data_size(Nrrd *);



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

template <class ImageType>
bool
Painter::do_itk_filter(itk::ImageToImageFilter<ImageType, ImageType> *filter,
                       NrrdDataHandle &nrrd_handle) 
{
  typedef typename itk::MemberCommand< Painter > RedrawCommandType;
  typename RedrawCommandType::Pointer callback = RedrawCommandType::New();
  callback->SetCallbackFunction(this, &Painter::filter_callback);
  callback->SetCallbackFunction(this, &Painter::filter_callback_const);
  filter->AddObserver(itk::ProgressEvent(), callback);
  filter->AddObserver(itk::IterationEvent(), callback);
  
  if (nrrd_handle.get_rep()) {
    ITKDatatypeHandle img_handle = nrrd_to_itk_image(nrrd_handle);  
    ImageType *imgp = dynamic_cast<ImageType *>(img_handle->data_.GetPointer());
    
    if (imgp == 0) 
      return false;
    
    filter->SetInput(imgp);
  }

  try {
    filter->Update();
  } catch (itk::ExceptionObject &err) {
    if (!abort_filter_) {
      cerr << "ITK Exception: \n";
      err.Print(cerr);
      return false;
    } else {
      abort_filter_ = false;
    }
  } catch (...) {
    cerr << "ITK Filter error!\n";
    return false;
  }
  
  SCIRun::ITKDatatypeHandle output_img = new SCIRun::ITKDatatype();
  output_img->data_ = filter->GetOutput();

  nrrd_handle = itk_image_to_nrrd(output_img);

#if 0
  get_vars()->insert("ProgressBar::bar_height","0","string", true);
  get_vars()->insert("Painter::progress_bar_total_width","0","string", true);
  get_vars()->insert("Painter::progress_bar_text","F","string", true);
  get_vars()->insert("Painter::progress_bar_done_width","0","string", true);
  get_vars()->insert("ToolDialog::button_height","0","string", true);
  get_vars()->insert("ToolDialog::text","","string", true);
#endif

  return true;
}
#endif

}
#endif
