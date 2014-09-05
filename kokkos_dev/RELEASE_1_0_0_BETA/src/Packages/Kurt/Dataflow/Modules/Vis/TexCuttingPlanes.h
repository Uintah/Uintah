#ifndef TEXCUTTINGPLANES_H
#define TEXCUTTINGPLANES_H
/*
 * TexCuttingPlanes.cc
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

#include <Packages/Kurt/Core/Datatypes/GLVolumeRenderer.h>
#include <Packages/Kurt/Core/Datatypes/GLTexture3DPort.h>
#include <Packages/Kurt/Core/Datatypes/GLTexture3D.h>

namespace Kurt {

using namespace SCIRun;

class TexCuttingPlanes : public Module {

public:
  TexCuttingPlanes( const clString& id);

  virtual ~TexCuttingPlanes();
  virtual void widget_moved(int last);    
  virtual void execute();
  void tcl_command( TCLArgs&, void* );

private:

  GLTexture3DHandle tex;

  ColorMapIPort* incolormap;
  GLTexture3DIPort* intexture;
  GeometryOPort* ogeom;
   
  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;

  int cmap_id;  // id associated with color map...

  GuiInt drawX;
  GuiInt drawY;
  GuiInt drawZ;
  GuiInt drawView;
  GuiInt interp_mode;

  GLVolumeRenderer* volren;
  Vector ddv;
  double ddview;
};

} // End namespace Kurt

#endif
