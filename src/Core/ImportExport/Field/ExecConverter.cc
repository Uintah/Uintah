/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

#include <Core/ImportExport/Field/FieldIEPlugin.h>
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
    result = 0;
  }
#else
  string command = icommand + " > " + tmpfilename + ".log 2>&1";
  const int status = sci_system(command.c_str());
  if (status != 0)
  {
    cerr << "ExecConverter syscal error " << status << ": "
	 << "command was '" << command << "'" << endl;
    result = 0;
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


static FieldHandle
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
    FieldHandle field;
    Pio(*stream, field);

    cerr << "ExecConverter - Successfully converted " << cfilename << endl;

    unlink(tmpfilename.c_str());
    
    return field;
  }

  unlink(tmpfilename.c_str());
  return 0;
}



static void
Exec_writer(ProgressReporter *pr,
	    FieldHandle field, const char *cfilename, const string &precommand)
{
  string command, tmpfilename;
  Exec_setup_command(cfilename, precommand, command, tmpfilename);

  Piostream *stream = scinew BinaryPiostream(tmpfilename, Piostream::Write);
  if (stream->error())
  {
    cerr << "Could not open temporary file '" + tmpfilename +
      "' for writing." << endl;
  }
  else
  {
    Pio(*stream, field);
    
    Exec_execute_command(command, tmpfilename);
    
  }
  unlink(tmpfilename.c_str());
  delete stream;
}



// CurveField

static FieldHandle
TextCurveField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToCurveFieldToText %e.pts %e.edges %t";
  return Exec_reader(pr, filename, command);
}

static void
TextCurveField_writer(ProgressReporter *pr,
		      FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "CurveFieldToText %t %e.pts %e.edges";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextCurveField_plugin("TextCurveField",
		      "", "",
		      TextCurveField_reader,
		      TextCurveField_writer);


// HexVolField

static FieldHandle
TextHexVolField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToHexVolField %e.pts %e.hexes %t";
  return Exec_reader(pr, filename, command);
}

static void
TextHexVolField_writer(ProgressReporter *pr,
		       FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "HexVolFieldToText %t %e.pts %e.hexes";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextHexVolField_plugin("TextHexVolField",
		       "", "",
		       TextHexVolField_reader,
		       TextHexVolField_writer);


// QuadSurfField

static FieldHandle
TextQuadSurfField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToQuadSurfField %e.pts %e.quads %t";
  return Exec_reader(pr, filename, command);
}

static void
TextQuadSurfField_writer(ProgressReporter *pr,
			 FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "QuadSurfFieldToText %t %e.pts %e.quads";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextQuadSurfField_plugin("TextQuadSurfField",
			 "", "",
			 TextQuadSurfField_reader,
			 TextQuadSurfField_writer);


// TetVolField

static FieldHandle
TextTetVolField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToTetVolField %e.pts %e.tets %t";
  return Exec_reader(pr, filename, command);
}

static void
TextTetVolField_writer(ProgressReporter *pr,
		       FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TetVolFieldToText %t %e.pts %e.tets";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextTetVolField_plugin("TextTetVolField",
		       "", "",
		       TextTetVolField_reader,
		       TextTetVolField_writer);


// TriSurfField

static FieldHandle
TextTriSurfField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToTriSurfField %e.pts %e.tets %t";
  return Exec_reader(pr, filename, command);
}

static void
TextTriSurfField_writer(ProgressReporter *pr,
			FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TriSurfFieldToText %t %e.pts %e.tets";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextTriSurfField_plugin("TextTriSurfField",
			"", "",
			TextTriSurfField_reader,
			TextTriSurfField_writer);





static FieldHandle
TextPointCloudField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToPointCloudField %f %t";
  return Exec_reader(pr, filename, command);
}

static void
TextPointCloudField_writer(ProgressReporter *pr,
			   FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "PointCloudFieldToText %t %f";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextPointCloudField_plugin("TextPointCloudField",
			   "", "",
			   TextPointCloudField_reader,
			   TextPointCloudField_writer);



static FieldHandle
TextStructCurveField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToStructCurveField %f %t";
  return Exec_reader(pr, filename, command);
}

static void
TextStructCurveField_writer(ProgressReporter *pr,
			    FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "StructCurveFieldToText %t %f";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextStructCurveField_plugin("TextStructCurveField",
			    "", "",
			    TextStructCurveField_reader,
			    TextStructCurveField_writer);



static FieldHandle
TextStructHexVolField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToStructHexVolField %f %t";
  return Exec_reader(pr, filename, command);
}

static void
TextStructHexVolField_writer(ProgressReporter *pr,
			     FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "StructHexVolFieldToText %t %f";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextStructHexVolField_plugin("TextStructHexVolField",
			     "", "",
			     TextStructHexVolField_reader,
			     TextStructHexVolField_writer);


static FieldHandle
TextStructQuadSurfField_reader(ProgressReporter *pr, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "TextToStructQuadSurfField %f %t";
  return Exec_reader(pr, filename, command);
}

static void
TextStructQuadSurfField_writer(ProgressReporter *pr,
			       FieldHandle field, const char *filename)
{
  const string command =
    string(SCIRUN_OBJDIR) + "/StandAlone/convert/" +
    "StructQuadSurfFieldToText %t %f";
  Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextStructQuadSurfField_plugin("TextStructQuadSurfField",
			       "", "",
			       TextStructQuadSurfField_reader,
			       TextStructQuadSurfField_writer);
