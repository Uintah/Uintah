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
 *  Painter.cc
 *
 *  Written by:
 *   McKay Davis
 *   School of Computing
 *   University of Utah
 *   November, 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <StandAlone/Apps/Painter/Painter.h>
#include <StandAlone/Apps/Painter/PointerToolSelectorTool.h>
#include <StandAlone/Apps/Painter/KeyToolSelectorTool.h>
#include <sci_comp_warn_fixes.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <sci_algorithm.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/ColorMappedNrrdTextureObj.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Skinner/GeomSkinnerVarSwitch.h>
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
#include <Core/Geom/GeomColorMappedNrrdTextureObj.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Util/FileUtils.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>


namespace SCIRun {



Painter::Painter(Skinner::Variables *variables, VarContext* ctx) :
  Parent(variables),
  cur_window_(0),
  tm_("Painter"),
  pointer_pos_(),
  windows_(),
  volumes_(),
  current_volume_(0),
  undo_volume_(0),
  colormaps_(),
  anatomical_coordinates_(0, 1),
  volume_lock_("Volume"),
  volume_texture_(0),
  filter_volume_(0),
  abort_filter_(false),
  status_(variables, "Painter::status", "")
{
#ifdef HAVE_INSIGHT
  filter_update_img_ = 0;
#endif

  tm_.add_tool(new PointerToolSelectorTool(this), 50);
  tm_.add_tool(new KeyToolSelectorTool(this), 51);

  InitializeSignalCatcherTargets(0);
  Skinner::Signal *signal = new Skinner::Signal("LoadColorMap1D",
                                                this, get_vars());
  string path = findFileInPath("Rainbow.cmap", sci_getenv("SKINNER_PATH"));
  if (!path.empty()) {
    Skinner::Var<string> filename(get_vars(), "filename");
    filename = path + "Rainbow.cmap";
    LoadColorMap1D(signal);
  }
}

Painter::~Painter()
{
//   int count = 1;
//   for (NrrdVolumes::iterator iter = volumes_.begin(); 
//        iter != volumes_.end(); ++iter) {
//     string filename = "/tmp/painter-nrrd"+to_string(count++)+".nrrd";
//     cerr << "saving " << filename << std::endl;
//     nrrdSave(filename.c_str(),
//              (*iter)->nrrd_handle_->nrrd_,
//              0);
//   }


}




void
Painter::redraw_all()
{
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->redraw();
  }
}



ColorMapHandle
Painter::get_colormap(int id)
{
  if (!id) return 0;
  id = Clamp(id-1, 0, colormaps_.size());
  if (id < (int)colormaps_.size())
    return colormaps_[id];
  return 0;
}
    


void
Painter::extract_all_window_slices() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->extract_slices();
  }
}



void
Painter::get_data_from_layer_buttons()  
{
  for (unsigned int i = 0; i < layer_buttons_.size(); ++i) {
    LayerButton *button = layer_buttons_[i];
    NrrdVolume *volume = button->volume_;
    if (!volume) continue;
    volume->name_ = button->layer_name_;
    volume->visible_ = button->layer_visible_;
    volume->expand_ = button->expand_;
  }
}

void
Painter::rebuild_layer_buttons()  
{
  get_data_from_layer_buttons();
  unsigned int bpos = 0;  
  for (int i = volumes_.size()-1; i >= 0 ; --i) {
    build_layer_button(bpos, volumes_[i]);
  }
  for (; bpos < layer_buttons_.size(); ++bpos) {
    layer_buttons_[bpos]->visible_ = false;
    layer_buttons_[bpos]->volume_ = 0;
  }
  EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
}

void
Painter::build_layer_button(unsigned int &bpos, NrrdVolume *volume)  
{
  LayerButton *button = layer_buttons_[bpos];
  button->visible_ = true;
  button->layer_name_ = volume->name_;
  button->volume_ = volume;

  unsigned int level = 0;
  NrrdVolume *parent = volume;
  while (parent->parent_) {
    level++;
    parent = parent->parent_;
  }
  button->indent_ = 20*level+5;
  if (volume->children_.empty()) {
    button->expand_width_ = 0;
  } else {
    button->expand_width_ = 20;
  }
    

  if (volume == current_volume_) {
    button->background_color_ = Skinner::Color(0.6, 0.6, 1.0, 0.75);
  } else {
    button->background_color_ = Skinner::Color(0.0, 0.0, 0.0, 0.0);
  }

  bpos++;  
  if (volume->expand_) {
    for (int i = volume->children_.size()-1; i >= 0 ; --i) {
      build_layer_button(bpos, volume->children_[i]);
    }
  }
}



void
Painter::build_volume_list(NrrdVolumes &volumes, NrrdVolume *volume)
{
  NrrdVolumes &children = volume ? volume->children_ : volumes_;
  for (unsigned int i = 0; i < children.size(); ++i) {
    volumes.push_back(children[i]);
    build_volume_list(volumes, children[i]);
  }
}

void
Painter::move_layer_up(NrrdVolume *layer)
{
  if (!layer) return;
  NrrdVolumes &volumes = layer->parent_ ? layer->parent_->children_ : volumes_;
  unsigned int i = 0;
  while (i < volumes.size() && volumes[i] != layer) ++i;
  ASSERT(volumes[i] == layer);
  if (i == volumes.size()-1) return;

  NrrdVolume *temp = volumes[i+1];
  volumes[i+1] = volumes[i];
  volumes[i] = temp;
  
  extract_all_window_slices();
  rebuild_layer_buttons();
}

void
Painter::move_layer_down(NrrdVolume *layer)
{
  if (!layer) return;  
  NrrdVolumes &volumes = layer->parent_ ? layer->parent_->children_ : volumes_;
  unsigned int i = 0;
  while (i < volumes.size() && volumes[i] != layer) ++i;
  ASSERT(volumes[i] == layer);
  if (i == 0) return;

  NrrdVolume *temp = volumes[i-1];
  volumes[i-1] = volumes[i];
  volumes[i] = temp;
  
  extract_all_window_slices();
  rebuild_layer_buttons();
}


void
Painter::opacity_down()
{
  if (current_volume_) {
    current_volume_->opacity_ = 
      Clamp(current_volume_->opacity_-0.05, 0.0, 1.0);
    redraw_all();
  }
}

void
Painter::opacity_up()
{
  if (current_volume_) {
    current_volume_->opacity_ = 
      Clamp(current_volume_->opacity_+0.05, 0.0, 1.0);
    redraw_all();
  }
}



void
Painter::cur_layer_down()
{
  if (volumes_.size() < 2 || current_volume_ == volumes_[0]) 
    return;
  for (unsigned int i = 1; i < volumes_.size(); ++i)
    if (current_volume_ == volumes_[i]) {
      current_volume_ = volumes_[i-1];
      rebuild_layer_buttons();
      return;
    }
}


void
Painter::cur_layer_up()
{
  if (volumes_.size() < 2 || current_volume_ == volumes_.back()) 
    return;
  for (unsigned int i = 0; i < volumes_.size()-1; ++i)
    if (current_volume_ == volumes_[i]) {
      current_volume_ = volumes_[i+1];
      rebuild_layer_buttons();
      return;
    }
}


void
Painter::reset_clut()
{
  if (current_volume_) {
    current_volume_->clut_min_ = current_volume_->data_min_;
    current_volume_->clut_max_ = current_volume_->data_max_;
    set_all_slices_tex_dirty();
    redraw_all();
  }
}



void
Painter::set_probe() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    (*i)->set_probe();
  }
  redraw_all();
}


void
Painter::set_all_slices_tex_dirty() {
  for (SliceWindows::iterator i = windows_.begin(); i != windows_.end(); ++i) {
    for (VolumeSlices::iterator s = (*i)->slices_.begin();
         s != (*i)->slices_.end(); ++s) {
      (*s)->set_tex_dirty();
    }
  }
}


NrrdVolume *
Painter::find_volume_by_name(const string &name) {
  for (unsigned int i = 0; i < volumes_.size(); ++i)
    if (volumes_[i]->name_ == name) 
      return volumes_[i];
  return 0;
}


void
Painter::copy_current_layer() {
  if (current_volume_) {
    string base = current_volume_->name_;
    string::size_type pos = base.find_last_not_of(" 0123456789");
    base = base.substr(0, pos+1);
    int i = 0;
    string name = base + " "+to_string(++i);
    while (find_volume_by_name(name))
      name = base + " "+to_string(++i);
    current_volume_ = copy_current_volume(name,0);
    rebuild_layer_buttons();
    extract_all_window_slices();
    redraw_all();

  }
}


void
Painter::new_current_layer() {
  if (current_volume_) {
    string base = "New Layer";
    int i = 0;
    string name = base + " "+to_string(++i);
    while (find_volume_by_name(name))
      name = base + " "+to_string(++i);
    current_volume_ = copy_current_volume(name,1);
    rebuild_layer_buttons();
    extract_all_window_slices();
    redraw_all();
  }
}


void
Painter::create_undo_volume() {
  return;
  if (undo_volume_) 
    delete undo_volume_;
  string newname = current_volume_->name_;
  undo_volume_ = scinew NrrdVolume(current_volume_, newname, 0);
}

void
Painter::undo_volume() {
#if 0
  if (!undo_volume_) return;
  NrrdVolume *vol = volume_map_[undo_volume_->name_];
  if (!vol) return;
  vol->nrrd_handle_ = undo_volume_->nrrd_handle_;
  vol->nrrd_handle_.detach();
  extract_all_window_slices();
  redraw_all();
#endif
  //  delete undo_volume_;
  //  undo_volume_ = 0;
}


pair<double, double>
Painter::compute_mean_and_deviation(Nrrd *nrrd, Nrrd *mask) {
  double mean = 0;
  double squared = 0;
  unsigned int n = 0;
  ASSERT(nrrd->dim > 3 && mask->dim > 3 && 
         nrrd->axis[0].size == mask->axis[0].size &&
         nrrd->axis[1].size == mask->axis[1].size &&
         nrrd->axis[2].size == mask->axis[2].size &&
         nrrd->axis[3].size == mask->axis[3].size &&
         nrrd->type == nrrdTypeFloat &&
         mask->type == nrrdTypeFloat);

  unsigned int size = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a)
    size *= nrrd->axis[a].size;

  float *src = (float *)nrrd->data;
  float *test = (float *)mask->data;

  float min = AIR_POS_INF;
  float max = AIR_NEG_INF;
  
  for (unsigned int i = 0; i < size; ++i)
    if (test[i] > 0.0) {
      //      cerr << test[i] << std::endl;
      mean += src[i];
      squared += src[i]*src[i];
      min = Min(min, src[i]);
      max = Max(max, src[i]);

      ++n;
    }

  mean = mean / n;
  double deviation = sqrt(squared/n-mean*mean);
  //  cerr << "size: " << size << " n: " << n << std::endl;
  //  cerr << "mean: " << mean << " dev: " << deviation << std::endl;
  //  return make_pair(min,max);
  return make_pair(mean, deviation);
}
  
         

NrrdVolume *
Painter::copy_current_volume(const string &name, int mode) {
  if (!current_volume_) return 0;
  NrrdVolume *vol = new NrrdVolume(current_volume_, name, mode);
  volumes_.push_back(vol);
  vol->clut_min_ = vol->data_max_/255.0;
  vol->clut_max_ = vol->data_max_;
  rebuild_layer_buttons();
  extract_all_window_slices();
  redraw_all();
  return vol;
}
  




#ifdef HAVE_INSIGHT

template <class PixType>
itk::Object::Pointer
cast_nrrd_to_itk(Nrrd *n) 
{
  const unsigned int dimension = 3;

  typedef itk::ImportImageFilter < PixType , 3 > ImportFilterType;

  typename ImportFilterType::Pointer importFilter = ImportFilterType::New();
  typename ImportFilterType::SizeType size;

  double origin[dimension];
  double spacing[dimension];
  unsigned int count = 1;
  for(unsigned int i=0; i < n->dim-1; i++) {
    count *= n->axis[i+1].size;
    size[i] = n->axis[i+1].size;

    if (!AIR_EXISTS(n->axis[i+1].min)) {
      origin[i] = 0;
    } else {
      origin[i] = n->axis[i+1].min;
    }

    if (!AIR_EXISTS(n->axis[i+1].spacing)) {
      spacing[i] = 1.0;
    } else {
      spacing[i] = n->axis[i+1].spacing;
    }
  }
  typename ImportFilterType::IndexType start;
  start.Fill(0);
  typename ImportFilterType::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);
  importFilter->SetRegion(region);
  importFilter->SetOrigin(origin);
  importFilter->SetSpacing(spacing);
  importFilter->SetImportPointer((PixType *)n->data, count, false);
  importFilter->Update();

  return importFilter->GetOutput();
}

ITKDatatypeHandle
Painter::nrrd_to_itk_image(NrrdDataHandle &nrrd) {
  Nrrd *n = nrrd->nrrd_;
  itk::Object::Pointer data = 0;
  switch (n->type) {
  case nrrdTypeChar: data = cast_nrrd_to_itk<signed char>(n); break;
  case nrrdTypeUChar: data = cast_nrrd_to_itk<unsigned char>(n); break;
  case nrrdTypeShort: data = cast_nrrd_to_itk<signed short>(n); break;
  case nrrdTypeUShort: data = cast_nrrd_to_itk<unsigned short>(n); break;
  case nrrdTypeInt: data = cast_nrrd_to_itk<signed int>(n); break;
  case nrrdTypeUInt: data = cast_nrrd_to_itk<unsigned int>(n); break;
  case nrrdTypeLLong: data = cast_nrrd_to_itk<signed long long>(n); break;
  case nrrdTypeULLong: data =cast_nrrd_to_itk<unsigned long long>(n); break;
  case nrrdTypeFloat: data = cast_nrrd_to_itk<float>(n); break;
  case nrrdTypeDouble: data = cast_nrrd_to_itk<double>(n); break;
  default: throw "nrrd_to_itk_image, cannot convert type" + to_string(n->type);
  }

  SCIRun::ITKDatatype *result = new SCIRun::ITKDatatype();
  result->data_ = data;
  return result;
}






void
Painter::filter_callback(itk::Object *object,
                         const itk::EventObject &event)
{
  itk::ProcessObject::Pointer process = 
    dynamic_cast<itk::ProcessObject *>(object);
  ASSERT(process);
  double value = process->GetProgress();
  if (typeid(itk::ProgressEvent) == typeid(event))
  {

    
    // progress bar is broken!

//     double total = get_vars()->get_double("Painter::progress_bar_total_width");
//     get_vars()->insert("Painter::progress_bar_done_width", 
//                        to_string(value * total), "string", true);
    
//     get_vars()->insert("Painter::progress_bar_text", 
//                        to_string(round(value * 100))+ " %  ", "string", true);

    if (filter_volume_ && filter_update_img_.get_rep()) {
      //      typedef Painter::ITKImageFloat3D ImageType;
      //      typedef itk::ImageToImageFilter<ImageType, ImageType> FilterType;
      typedef itk::ThresholdSegmentationLevelSetImageFilter
        < ITKImageFloat3D, ITKImageFloat3D > FilterType;
      
      
      FilterType *filter = dynamic_cast<FilterType *>(object);
      ASSERT(filter);
      volume_lock_.lock();
      filter_update_img_->data_ = filter->GetOutput();
      //filter_update_img_->data_ = filter->GetFeatureImage();
      filter_volume_->nrrd_handle_ = itk_image_to_nrrd<float>(filter_update_img_);
      volume_lock_.unlock();
      if (volume_texture_.get_rep()) {
        NrrdTextureBuilderAlgo::build_static(volume_texture_,
                                             current_volume_->nrrd_handle_, 0, 255,
                                             0, 0, 255, 128);
      }

      //        volume_texture_->set_dirty(true);

      set_all_slices_tex_dirty();
      redraw_all();
    }

    redraw_all();

  }



  if (typeid(itk::IterationEvent) == typeid(event))
  {
    //    std::cerr << "Filter Iteration: " << value * 100.0 << "%\n";
  }
  if (abort_filter_) {
    //    abort_filter_ = false;
    process->AbortGenerateDataOn();
  }


}

void
Painter::filter_callback_const(const itk::Object *object,
                               const itk::EventObject &event)
{
  itk::ProcessObject::ConstPointer process = 
    dynamic_cast<const itk::ProcessObject *>(object);
  ASSERT(process);
  double value = process->GetProgress();
  if (typeid(itk::ProgressEvent) == typeid(event))
  {
    std::cerr << "Const Filter Progress: " << value * 100.0 << "%\n";
  }

  if (typeid(itk::IterationEvent) == typeid(event))
  {
    std::cerr << "Const Filter Iteration: " << value * 100.0 << "%\n";
  }
}
#endif // HAVE_INSIGHT


int
Painter::get_signal_id(const string &signalname) const {
  if (signalname == "SliceWindow_Maker") return 1;
  if (signalname == "LayerButton_Maker") return 2;
  if (signalname == "Painter::start_brush_tool") return 3;
  return 0;
}



Skinner::Drawable *
Painter::maker(Skinner::Variables *vars) 
{
  return new Painter(vars, 0);
}




  


#if 0

#include <Core/Volume/VolumeRenderer.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Volume/ColorMap2.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>

void
setup_volume_rendering() {
  event_handle_t scene_event = 0;
  
  CompileInfoHandle ci =
    NrrdTextureBuilderAlgo::get_compile_info(nrrd->type,nrrd->type);
  
  
  const int card_mem = 128;
  cerr << "nrrd texture\n";
  TextureHandle texture = new Texture;
  NrrdTextureBuilderAlgo::build_static(texture, 
				       nrrd_handle, 0, 255,
				       0, 0, 255, card_mem);
  vector<Plane *> *planes = new vector<Plane *>;
  
  
  string fn = string(argv[3]);
  Piostream *stream = auto_istream(fn, 0);
  if (!stream) {
    cerr << "Error reading file '" + fn + "'." << std::endl;
    return -1;
  }  
  // read the file.
  ColorMap2 *cmap2 = new ColorMap2();
  ColorMap2Handle icmap = cmap2;
  try {
    Pio(*stream, icmap);
  } catch (...) {
    cerr << "Error loading "+fn << std::endl;
    icmap = 0;
  }
  delete stream;
  ColorMapHandle cmap;
  vector<ColorMap2Handle> *cmap2v = new vector<ColorMap2Handle>(0);
  cmap2v->push_back(icmap);
  string enabled("111111");
  if (sci_getenv("CMAP_WIDGETS")) 
    enabled = sci_getenv("CMAP_WIDGETS");
  for (unsigned int i = 0; i < icmap->widgets().size(); ++ i) {
    if (i < enabled.size() && enabled[i] == '1') {
      icmap->widgets()[i]->set_onState(1); 
    } else {
      icmap->widgets()[i]->set_onState(0); 
    }
  }

  VolumeRenderer *vol = new VolumeRenderer(texture, 
					   cmap, 
					   *cmap2v, 
					   *planes,
					   Round(card_mem*1024*1024*0.8));
  vol->set_slice_alpha(-0.5);
  vol->set_interactive_rate(3.0);
  vol->set_sampling_rate(3.0);
  vol->set_material(0.322, 0.868, 1.0, 18);
  scene_event = new SceneGraphEvent(vol, "FOO");  
  //  if (!sci_getenv_p("PAINTER_NOSCENE")) 
  //    EventManager::add_event(scene_event);    

}  



#endif


} // end namespace SCIRun
