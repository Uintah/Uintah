#ifndef TEXTUREVOLVIS_H
#define TEXTUREVOLVIS_H
/*
 * TextureVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Core/Datatypes/GLVolumeRenderer.h>
#include <Core/Datatypes/GLTexture3D.h>
#include <Dataflow/Ports/GLTexture3DPort.h>


namespace SCIRun {

class GeomObj;


class TextureVolVis : public Module {

public:
  TextureVolVis( const clString& id);

  virtual ~TextureVolVis();
  virtual void widget_moved(int last);    
  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );

private:
  
  GLTexture3DHandle tex;

  ColorMapIPort* incolormap;
  GLTexture3DIPort* intexture;
  GeometryOPort* ogeom;
   
  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


  int cmap_id;  // id associated with color map...
  

  GuiInt num_slices;
  GuiInt draw_mode;
  GuiInt render_style;
  GuiDouble alpha_scale;
  GuiInt interp_mode;
  GLVolumeRenderer *volren;



  void SwapXZ( /*FieldHandle sfh*/ );
};

} // End namespace SCIRun

#endif
