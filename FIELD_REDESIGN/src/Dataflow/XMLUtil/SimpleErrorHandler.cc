
#include <PSECore/XMLUtil/SimpleErrorHandler.h>
#include <PSECore/XMLUtil/XMLUtil.h>
#include <iostream>
#ifndef __sgi
#include <stdio.h>
#endif
using namespace std;
using PSECore::XMLUtil::SimpleErrorHandler;

SimpleErrorHandler::SimpleErrorHandler()
{
    foundError=false;
}

SimpleErrorHandler::~SimpleErrorHandler()
{
}

static string to_string(int i)
{
    char buf[20];
    sprintf(buf, "%d", i);
    return string(buf);
}

static void postMessage(const string& errmsg)
{
    cerr << errmsg << '\n';
}

void SimpleErrorHandler::error(const SAXParseException& e)
{
    foundError=true;
    postMessage(string("Error at (file ")+toString(e.getSystemId())
		+", line "+toString((int)e.getLineNumber())
		+", char "+toString((int)e.getColumnNumber())
		+"): "+toString(e.getMessage()));
}

void SimpleErrorHandler::fatalError(const SAXParseException& e)
{
    foundError=true;
    postMessage(string("Fatal Error at (file ")+toString(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+toString(e.getMessage()));
}

void SimpleErrorHandler::warning(const SAXParseException& e)
{
    postMessage(string("Warning at (file ")+toString(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+toString(e.getMessage()));
}

void SimpleErrorHandler::resetErrors()
{
}

//
// $Log$
// Revision 1.1.2.2  2000/10/26 14:16:59  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.2  2000/06/11 19:09:18  moulding
// added #include <stdio.h> for sprintf()
//
// Revision 1.1  2000/05/20 08:04:28  sparker
// Added XML helper library
//
//
