/*
 *  ImageFileReader.h
 *  
 *  Written by:
 *   darbyb
 *   Dec 31, 2002
 *
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Insight/share/share.h>

#include <Insight/Dataflow/Ports/ITKImagePort.h>

#include "itkImageFileReader.h"

namespace Insight {

using namespace SCIRun;
 
typedef itk::ImageFileReader<ShortImageType> ReaderType;
typedef itk::PNGImageIO IOType;

class InsightSHARE ImageFileReader : public Module {
public:

  //! GUI variables
  GuiString gui_filename_;

  ReaderType::Pointer reader_;
  IOType::Pointer io_;
  ITKImageOPort* outport_;
  ITKImageHandle handle_;
  
  ImageFileReader(GuiContext*);

  virtual ~ImageFileReader();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};

} // End namespace Insight


