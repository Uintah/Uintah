/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
 *  ExecConverter.cc
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   December 2004
 *
 *  Copyright (C) 2004 SCI Institute
 */

// Use a standalone converter to do the colormap conversion into a
// temporary file, then read in that file.

#include <Core/Malloc/Allocator.h>
#include <Core/ImportExport/ColorMap/ColorMapIEPlugin.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/sci_system.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace std;
using namespace SCIRun;


static void
Exec_setup_command(const char *cfilename, const string &precommand,
		   string &command, string &tmpfilename)
{
  // Filename as string
  const string filename(cfilename);
  
  // Base filename.
  string::size_type loc = filename.find_last_of("/");
  const string basefilename =
    (loc==string::npos)?filename:filename.substr(loc+1);

  // Base filename with first extension removed.
  loc = basefilename.find_last_of(".");
  const string basenoext = basefilename.substr(0, loc);

  // Filename with first extension removed.
  loc = filename.find_last_of(".");
  const string noext = filename.substr(0, loc);

  // Temporary filename.
  tmpfilename = "/tmp/" + basenoext + "-" +
    to_string((unsigned int)(getpid())) + ".cmap";

  // Filename with first extension removed.
  loc = filename.find_last_of(".");
  const string filenoext = filename.substr(0, loc);

  // Replace all of the variables in the reader command.
  command = precommand;
  while ((loc = command.find("%f")) != string::npos)
  {
    command.replace(loc, 2, filename);
  }
  while ((loc = command.find("%e")) != string::npos)
  {
    command.replace(loc, 2, noext);
  }
  while ((loc = command.find("%t")) != string::npos)
  {
    command.replace(loc, 2, tmpfilename);
  }
}


static bool
Exec_execute_command(ProgressReporter *pr,
		     const string &icommand, const string &tmpfilename)
{
  pr->remark("ExecConverter - Executing: " + icommand + ".");

  FILE *pipe = 0;
  bool result = true;
#ifdef __sgi
  string command = icommand + " 2>&1";
  pipe = popen(command.c_str(), "r");
  if (pipe == NULL)
  {
    pr->error("ExecConverter syscal error, command was: '" + command + ".");
    result = false;
  }
#else
  string command = icommand + " > " + tmpfilename + ".log 2>&1";
  const int status = sci_system(command.c_str());
  if (status != 0)
  {
    pr->error("ExecConverter syscal error " + to_string(status) + ": "
	      + "command was '" + command + "'.");
    result = false;
  }
  pipe = fopen((tmpfilename + ".log").c_str(), "r");
#endif

  char buffer[256];
  while (pipe && fgets(buffer, 256, pipe) != NULL)
  {
    pr->msgStream() << buffer;
  }

#ifdef __sgi
  if (pipe) { pclose(pipe); }
#else
  if (pipe)
  {
    fclose(pipe);
    unlink((tmpfilename + ".log").c_str());
  }
#endif

  return result;
}


static ColorMapHandle
Exec_reader(ProgressReporter *pr,
	    const char *cfilename, const string &precommand)
{
  string command, tmpfilename;
  Exec_setup_command(cfilename, precommand, command, tmpfilename);

  if (Exec_execute_command(pr, command, tmpfilename))
  {
    Piostream *stream = auto_istream(tmpfilename);
    if (!stream)
    {
      pr->error("ExecConverter - Error reading converted file '" +
		tmpfilename + "'.");
      return 0;
    }
    
    // Read the file
    ColorMapHandle colormap;
    Pio(*stream, colormap);

    pr->remark(string("ExecConverter - Successfully converted ")
	       + cfilename + ".");

    unlink(tmpfilename.c_str());
    
    return colormap;
  }

  unlink(tmpfilename.c_str());
  return 0;
}



static bool
Exec_writer(ProgressReporter *pr,
	    ColorMapHandle colormap,
	    const char *cfilename, const string &precommand)
{
  string command, tmpfilename;
  bool result = true;

  Exec_setup_command(cfilename, precommand, command, tmpfilename);

  Piostream *stream = scinew BinaryPiostream(tmpfilename, Piostream::Write);
  if (stream->error())
  {
    pr->error("ExecConverter - Could not open temporary file '" + tmpfilename +
	      "' for writing.");
    result = false;
  }
  else
  {
    Pio(*stream, colormap);
    
    result = Exec_execute_command(pr, command, tmpfilename);
  }
  unlink(tmpfilename.c_str());
  delete stream;

  return result;
}



// CurveColorMap

///// NOTE: The following 2 procedures are not static because I need to
/////       reference them in FieldIEPlugin.cc to force the Mac OSX to 
/////       instantiate static libraries.

ColorMapHandle
TextColorMap_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToColorMap %f %t";
  return Exec_reader(pr, filename, command);
}

bool
TextColorMap_writer(ProgressReporter *pr,
		    ColorMapHandle colormap, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "ColorMapToText %f %t";
  return Exec_writer(pr, colormap, filename, command);
}

#ifndef __APPLE__
// On the Mac, this is done in FieldIEPlugin.cc, in the
// macImportExportForceLoad() function to force the loading of this
// (and other) plugins.
static ColorMapIEPlugin
TextColorMap_plugin("TextColorMap",
		    "", "",
		    TextColorMap_reader,
		    TextColorMap_writer);
#endif

