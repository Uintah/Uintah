#ifndef TEXTUREVOLVIS_H
#define TEXTUREVOLVIS_H
/*
 * TextureVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <PSECore/Widgets/PointWidget.h>

#include <Kurt/Datatypes/GLVolumeRenderer.h>
#include <Kurt/Datatypes/GLTexture3DPort.h>
#include <Kurt/Datatypes/GLTexture3D.h>

namespace Kurt {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace Kurt::GeomSpace;
using namespace SCICore::Geometry;
class SCICore::GeomSpace::GeomObj;


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
  

  TCLint num_slices;
  TCLint draw_mode;
  TCLint render_style;
  TCLdouble alpha_scale;
  TCLdouble alpha_gamma;
  TCLint interp_mode;
  GLVolumeRenderer *volren;



  void SwapXZ( ScalarFieldHandle sfh );
};

} // namespace Modules
} // namespace Uintah

#endif
