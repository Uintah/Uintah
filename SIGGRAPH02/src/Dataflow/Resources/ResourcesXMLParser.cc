

#include <Dataflow/Resources/ResourcesXMLParser.h>

namespace SCIRun {


ResourcesXMLParser::ResourcesXMLParser( Resources *s )
  : XMLParser(), resources_(s)
{
}

ResourcesXMLParser::~ResourcesXMLParser()
{
}

} // namespace SCIRun
