#include <PSECore/Dataflow/PackageDBHandler.h>

#include <SCICore/Containers/String.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <PSECore/Dataflow/NetworkEditor.h>

namespace PSECore {
namespace Dataflow {

using namespace SCICore::Containers;
using namespace PSECore::XMLUtil;

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

} // Dataflow
} // PSECore
