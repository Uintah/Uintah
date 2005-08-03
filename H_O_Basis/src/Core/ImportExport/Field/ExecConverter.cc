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

// Use a standalone converter to convert a scirun object into a
// temporary file, then read in that file.

#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <Core/ImportExport/ExecConverter.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/sci_system.h>
#include <Core/Util/Environment.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#ifdef _WIN32
#include <process.h>
#include <io.h>
#endif

using namespace std;
using namespace SCIRun;

namespace SCIRun {

void
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
    to_string((unsigned int)(getpid())) + ".sci";

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


bool
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
    pr->msgStream_flush();
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

} // namespace SCIRun



// CurveField

static FieldHandle
TextCurveField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToCurveField %e.pts %e.edge %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextCurveField_writer(ProgressReporter *pr,
		      FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "CurveFieldToText %t %e.pts %e.edge";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextCurveField_plugin("TextCurveField",
		      "{.pts} {.edge}", "",
		      TextCurveField_reader,
		      TextCurveField_writer);


// HexVolField

static FieldHandle
TextHexVolField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToHexVolField %e.pts %e.hex %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextHexVolField_writer(ProgressReporter *pr,
		       FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "HexVolFieldToText %t %e.pts %e.hex";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextHexVolField_plugin("TextHexVolField",
		       "{.pts} {.hex}", "",
		       TextHexVolField_reader,
		       TextHexVolField_writer);


// QuadSurfField

static FieldHandle
TextQuadSurfField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToQuadSurfField %e.pts %e.quad %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextQuadSurfField_writer(ProgressReporter *pr,
			 FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "QuadSurfFieldToText %t %e.pts %e.quad";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextQuadSurfField_plugin("TextQuadSurfField",
			 "{.pts} {.quad}", "",
			 TextQuadSurfField_reader,
			 TextQuadSurfField_writer);


// TetVolField

static FieldHandle
TextTetVolField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToTetVolField %e.pts %e.tet %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextTetVolField_writer(ProgressReporter *pr,
		       FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TetVolFieldToText %t %e.pts %e.tet";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextTetVolField_plugin("TextTetVolField",
		       "{.pts} {.tet}", "",
		       TextTetVolField_reader,
		       TextTetVolField_writer);


// TriSurfField

static FieldHandle
TextTriSurfField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToTriSurfField %e.pts %e.fac %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextTriSurfField_writer(ProgressReporter *pr,
			FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TriSurfFieldToText %t %e.pts %e.fac";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextTriSurfField_plugin("TextTriSurfField",
			"{.pts} {.fac} {.tri}", "",
			TextTriSurfField_reader,
			TextTriSurfField_writer);





static FieldHandle
TextPointCloudField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToPointCloudField %f %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextPointCloudField_writer(ProgressReporter *pr,
			   FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "PointCloudFieldToText %t %f";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextPointCloudField_plugin("TextPointCloudField",
			   ".pts", "",
			   TextPointCloudField_reader,
			   TextPointCloudField_writer);



static FieldHandle
TextStructCurveField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToStructCurveField %f %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextStructCurveField_writer(ProgressReporter *pr,
			    FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "StructCurveFieldToText %t %f";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextStructCurveField_plugin("TextStructCurveField",
			    ".pts", "",
			    TextStructCurveField_reader,
			    TextStructCurveField_writer);



static FieldHandle
TextStructHexVolField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToStructHexVolField %f %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextStructHexVolField_writer(ProgressReporter *pr,
			     FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "StructHexVolFieldToText %t %f";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextStructHexVolField_plugin("TextStructHexVolField",
			     ".pts", "",
			     TextStructHexVolField_reader,
			     TextStructHexVolField_writer);


static FieldHandle
TextStructQuadSurfField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToStructQuadSurfField %f %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static bool
TextStructQuadSurfField_writer(ProgressReporter *pr,
			       FieldHandle field, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "StructQuadSurfFieldToText %t %f";
  return Exec_writer(pr, field, filename, command);
}

static FieldIEPlugin
TextStructQuadSurfField_plugin("TextStructQuadSurfField",
			       ".pts", "",
			       TextStructQuadSurfField_reader,
			       TextStructQuadSurfField_writer);


// VTK Trisurf files.

static FieldHandle
VTKtoTriSurfFieldswap_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "VTKtoTriSurfField %f %t -swap_endian -bin_out";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static FieldIEPlugin
VTKtoTriSurfFieldswap_plugin("VTKtoTriSurfField-swap",
			     ".vtk", "",
			     VTKtoTriSurfFieldswap_reader,
			     NULL);


static FieldHandle
VTKtoTriSurfField_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "VTKtoTriSurfField %f %t -_endian -bin_out";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static FieldIEPlugin
VTKtoTriSurfField_plugin("VTKtoTriSurfField",
			     ".vtk", "",
			     VTKtoTriSurfField_reader,
			     NULL);

// WGET wrapper, example for fetching urls remotely.
static FieldHandle
wget_field_reader(ProgressReporter *pr, const char *filename)
{
  const string command = "wget -O %t %f";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static FieldIEPlugin
wget_field_plugin("WGET a SCIRun file",
		  ".fld", "",
		  wget_field_reader,
		  NULL);
			     
// Conversion of a tetrahedra FE mesh in VGRID *.gmv format into SCIRun *.pts/*.tet format
static FieldHandle
VgridTetGmv_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
   string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/gmvToPts -t %f %t1 && " +
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToTetVolField %t1.pts %t1.tet %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static FieldIEPlugin
VgridTetGmv_plugin("VgridTetGmv",
		       "{.gmv}", "",
		       VgridTetGmv_reader,
		       NULL);

static FieldHandle
// Conversion of a hexahedra FE mesh in VGRID *.gmv format into SCIRun *.pts/*.hex format
VgridHexGmv_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
   string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/gmvToPts -h %f %t1 && " +
    string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
    "TextToHexVolField %t1.pts %t1.hex %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static FieldIEPlugin
VgridHexGmv_plugin("VgridHexGmv",
		       "{.gmv}", "",
		       VgridHexGmv_reader,
		       NULL);

// Conversion of an tetrahedra FE-mesh in NeuroFEM/CAUCHY/CURRY-geo format into SCIRun *.pts/*.tet format
static FieldHandle
NeuroFEMTetGeo_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
   string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/geoToPts -t %f %t1 && " +
   string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
   "TextToTetVolField %t1.pts %t1.tet %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static FieldIEPlugin
NeuroFEMTetGeo_plugin("NeuroFEMTetGeo",
		       "{.geo}", "",
		       NeuroFEMTetGeo_reader,
		       NULL);

//  Conversion of a hexahedra FE-mesh in NeuroFEM/CAUCHY/CURRY-geo format into SCIRun *.pts/*.hex format
static FieldHandle
NeuroFEMHexGeo_reader(ProgressReporter *pr, const char *filename)
{
  ASSERT(sci_getenv("SCIRUN_OBJDIR"));
  const string command =
   string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/geoToPts -h %f %t1 && " +
   string(sci_getenv("SCIRUN_OBJDIR")) + "/StandAlone/convert/" +
   "TextToHexVolField %t1.pts %t1.hex %t -binOutput";
  FieldHandle result;
  Exec_reader(pr, result, filename, command);
  return result;
}

static FieldIEPlugin
NeuroFEMHexGeo_plugin("NeuroFEMHexGeo",
		       "{.geo}", "",
		       NeuroFEMHexGeo_reader,
		       NULL);

/*****************************************************************/

