
#include <Dataflow/XMLUtil/SimpleErrorHandler.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <iostream>
#ifndef __sgi
#include <stdio.h>
#endif
using namespace std;

namespace SCIRun {

SimpleErrorHandler::SimpleErrorHandler()
{
    foundError=false;
}

SimpleErrorHandler::~SimpleErrorHandler()
{
}

static string sehToString(int i)
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
		+", line "+sehToString((int)e.getLineNumber())
		+", char "+sehToString((int)e.getColumnNumber())
		+"): "+toString(e.getMessage()));
}

void SimpleErrorHandler::warning(const SAXParseException& e)
{
    postMessage(string("Warning at (file ")+toString(e.getSystemId())
		+", line "+sehToString((int)e.getLineNumber())
		+", char "+sehToString((int)e.getColumnNumber())
		+"): "+toString(e.getMessage()));
}

void SimpleErrorHandler::resetErrors()
{
}

} // End namespace SCIRun







