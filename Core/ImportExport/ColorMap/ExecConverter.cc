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

// Use a standalone converter to do the field conversion into a
// temporary file, then read in that file.

#include <Core/Malloc/Allocator.h>
#include <Core/ImportExport/ColorMap/ColorMapIEPlugin.h>
#include <Core/Persistent/Pstreams.h>
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
  const string basefilename = filename.substr(loc);

  // Base filename with first extension removed.
  loc = basefilename.find_last_of(".");
  const string basenoext = basefilename.substr(0, loc);

  // Temporary filename.
  tmpfilename = "/tmp/" + basenoext + "-" + "123456" + ".fld";

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
    command.replace(loc, 2, basenoext);
  }
  while ((loc = command.find("%t")) != string::npos)
  {
    command.replace(loc, 2, tmpfilename);
  }
}


static bool
Exec_execute_command(const string &icommand, const string &tmpfilename)
{
  cerr << "ExecConverter - Executing: " << icommand << endl;

  FILE *pipe = 0;
  bool result = true;
#ifdef __sgi
  string command = icommand + " 2>&1";
  pipe = popen(command.c_str(), "r");
  if (pipe == NULL)
  {
    cerr << "ExecConverter syscal error, command was: '" << command << "'\n";
    result = false;
  }
#else
  string command = icommand + " > " + tmpfilename + ".log 2>&1";
  const int status = sci_system(command.c_str());
  if (status != 0)
  {
    cerr << "ExecConverter syscal error " << status << ": "
	 << "command was '" << command << "'" << endl;
    result = false;
  }
  pipe = fopen((tmpfilename + ".log").c_str(), "r");
#endif

  char buffer[256];
  while (pipe && fgets(buffer, 256, pipe) != NULL)
  {
    cerr << buffer;
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
Exec_reader(ProgressReporter *pr, const char *cfilename, const string &precommand)
{
  string command, tmpfilename;
  Exec_setup_command(cfilename, precommand, command, tmpfilename);

  if (Exec_execute_command(command, tmpfilename))
  {
    Piostream *stream = auto_istream(tmpfilename);
    if (!stream)
    {
      cerr << "Error reading converted file '" + tmpfilename + "'." << endl;
      return 0;
    }
    
    // Read the file
    ColorMapHandle field;
    Pio(*stream, field);

    cerr << "ExecConverter - Successfully converted " << cfilename << endl;

    unlink(tmpfilename.c_str());
    
    return field;
  }

  unlink(tmpfilename.c_str());
  return 0;
}



static bool
Exec_writer(ProgressReporter *pr,
	    ColorMapHandle field, const char *cfilename, const string &precommand)
{
  string command, tmpfilename;
  bool result = true;

  Exec_setup_command(cfilename, precommand, command, tmpfilename);

  Piostream *stream = scinew BinaryPiostream(tmpfilename, Piostream::Write);
  if (stream->error())
  {
    cerr << "Could not open temporary file '" + tmpfilename +
      "' for writing." << endl;
    result = false;
  }
  else
  {
    Pio(*stream, field);
    
    result = Exec_execute_command(command, tmpfilename);
  }
  unlink(tmpfilename.c_str());
  delete stream;

  return result;
}



// CurveColorMap

static ColorMapHandle
TextColorMap_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToColorMap %f %t";
  return Exec_reader(pr, filename, command);
}

static bool
TextColorMap_writer(ProgressReporter *pr,
		    ColorMapHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "ColorMapToText %f %t";
  return Exec_writer(pr, field, filename, command);
}

static ColorMapIEPlugin
TextColorMap_plugin("TextColorMap",
		    "", "",
		    TextColorMap_reader,
		    TextColorMap_writer);
