#include <Dataflow/Network/PackageDBHandler.h>

#include <Core/Containers/String.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Dataflow/Network/NetworkEditor.h>

namespace SCIRun {


PackageDBHandler::PackageDBHandler()
{
  foundError=false;
}

PackageDBHandler::~PackageDBHandler()
{
}

void PackageDBHandler::error(const SAXParseException& e)
{
  foundError=true;
  postMessage(clString("Error at (file ")+xmlto_string(e.getSystemId())
	      +", line "+to_string((int)e.getLineNumber())
	      +", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::fatalError(const SAXParseException& e)
{
  foundError=true;
  postMessage(clString("Fatal Error at (file ")+xmlto_string(e.getSystemId())
	      +", line "+to_string((int)e.getLineNumber())
	      +", char "+to_string((int)e.getColumnNumber())
	      +"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::warning(const SAXParseException& e)
{
  postMessage(clString("Warning at (file ")+xmlto_string(e.getSystemId())
	      +", line "+to_string((int)e.getLineNumber())
	      +", char "+to_string((int)e.getColumnNumber())
	      +"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::resetErrors()
{
}

} // End namespace SCIRun
