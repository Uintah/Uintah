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
 *  ForwardIPM.cc:
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/ImportExport/ExecConverter.h>
#include <Core/Containers/StringUtil.h>

#include <sys/stat.h>
#include <sys/types.h>


namespace SCIRun {

class ForwardIPM : public Module {
public:
  ForwardIPM(GuiContext* ctx);

  virtual void execute();
};


DECLARE_MAKER(ForwardIPM)

ForwardIPM::ForwardIPM(GuiContext* ctx)
  : Module("ForwardIPM", ctx, Filter, "NeuroFEM", "BioPSE")
{
}


void
ForwardIPM::execute()
{
  FieldIPort *condmesh_port = (FieldIPort *)get_iport("CondMesh");
  FieldHandle condmesh;
  if (!(condmesh_port->get(condmesh) && condmesh.get_rep()))
  {
    warning("CondMesh field required to continue.");
    return;
  }

  if (condmesh->mesh()->get_type_description()->get_name() != "TetVolMesh")
  {
    error("CondMesh must contain a TetVolMesh.");
    return;
  }
  if (condmesh->query_tensor_interface(this) == 0)
  {
    error("CondMesh must be a tensor field.");
    return;
  }
  if (condmesh->basis_order() != 0)
  {
    error("CondMesh must have conductivities at the cell centers.");
    return;
  }

  FieldIPort *electrodes_port = (FieldIPort *)get_iport("Electrodes");
  FieldHandle electrodes;
  if (!(electrodes_port->get(electrodes) && electrodes.get_rep()))
  {
    warning("Electrodes field required to continue.");
    return;
  }
  
  FieldIPort *dipole_port = (FieldIPort *)get_iport("Dipole");
  FieldHandle dipole;
  if (!(dipole_port->get(dipole) && dipole.get_rep()))
  {
    warning("Dipole required to continue.");
    return;
  }

  if (false) // Some dipole error checking.
  {
    error("Dipole must contain Vectors at Nodes.");
    return;
  }

  // Make our tmp directory
  const string tmpdir = "/tmp/ForwardIPM" + to_string(getpid());
  mode_t umsk = umask(00);
  mkdir(tmpdir.c_str(), 0700);
  umask(umsk);

  const string tmplog = tmpdir + "forward.log";
  const string resultfile = tmpdir + "result.msr";

  // Write out condmesh
  const string condmeshfile = tmpdir + "condmesh.geo";
  const string condtensfile = tmpdir + "ca_perm.knw";

  // Write out Electrodes
  const string electrodefile = tmpdir + "electrode.elc";

  // Write out Dipole
  const string dipolefile = tmpdir + "dipole.dip";

  // Write out parameter file.
  const string parafile = tmpdir + "forward.par";

  // Construct our command line.
  const string impfile = "imp";
  const string command = "(cd tmpfile;" + impfile + " -i sourcesimulation" +
    " -h " + condmeshfile + " -p " + parafile + " -s " + electrodefile +
    " -d " + dipolefile + " -o " + resultfile + " -fwd FEM -sens EEG)";

  // Execute the command.  Last arg just a tmp file for logging.
  if (!Exec_execute_command(this, command, tmplog))
  {
    error("The ipm program failed to run for some unknown reason.");
    return;
  }

  // Clean up our temporary files.
  unlink(condmeshfile.c_str());
  unlink(condtensfile.c_str());
  unlink(parafile.c_str());
  unlink(electrodefile.c_str());
  unlink(dipolefile.c_str());
  unlink(resultfile.c_str());
  unlink(tmplog.c_str());
  rmdir(tmpdir.c_str());
}


} // End namespace SCIRun
