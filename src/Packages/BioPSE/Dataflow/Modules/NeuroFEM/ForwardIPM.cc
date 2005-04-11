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
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PointCloudField.h>


#include <sys/stat.h>
#include <sys/types.h>


namespace SCIRun {

class ForwardIPM : public Module {
public:
  ForwardIPM(GuiContext* ctx);

  virtual void execute();

private:

  bool write_par_file(string filename);
};


DECLARE_MAKER(ForwardIPM)

ForwardIPM::ForwardIPM(GuiContext* ctx)
  : Module("ForwardIPM", ctx, Filter, "NeuroFEM", "BioPSE")
{
}



static bool
write_geo_file(ProgressReporter *pr, FieldHandle field, const char *filename)
{
  TetVolMesh *mesh = dynamic_cast<TetVolMesh *>(field->mesh().get_rep());
  if (mesh == 0)
  {
    pr->error("Field does not contain a TetVolMesh.");
    return false;
  }

  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
    pr->error(string("Unable to open file '") + filename + "' for writing.");
    return false;
  }

  // Write header.
  
  // Write it out.
  TetVolMesh::Node::iterator nitr, neitr;
  mesh->begin(nitr);
  mesh->end(neitr);
  
  while (nitr != neitr)
  {
    Point p;
    mesh->get_point(p, *nitr);

    // Write p;

    ++nitr;
  }


  TetVolMesh::Cell::iterator citr, ceitr;
  mesh->begin(citr);
  mesh->end(ceitr);
  
  while (citr != ceitr)
  {
    TetVolMesh::Node::array_type nodes; // 0 based
    mesh->get_nodes(nodes, *citr);

    // Write nodes
    fprintf(f, "%d %d %d %d\n",
            nodes[0]+1, nodes[1]+1, nodes[2]+1, nodes[3]+1);

    ++citr;
  }

  // Write tail, bottom part.


  fclose(f);
  return true;
}



static bool
write_knw_file(ProgressReporter *pr, FieldHandle field, const char *filename)
{
  TetVolField<Tensor> *tfield =
    dynamic_cast<TetVolField<Tensor> *>(field.get_rep());
  TetVolMesh *mesh = dynamic_cast<TetVolMesh *>(field->mesh().get_rep());
  if (mesh == 0)
  {
    pr->error("Field does not contain a TetVolMesh.");
    return false;
  }

  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
    pr->error(string("Unable to open file '") + filename + "' for writing.");
    return false;
  }

  // Write header.
  
  // Write it out.
  TetVolMesh::Elem::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);
  
  while (itr != eitr)
  {
    Tensor t;
    tfield->value(t, *itr);

    fprintf(f, "    %d   %f    %f    %f\n          %f     %f      %f\n",
            1+*itr,  // fortran index, +1
            t.mat_[0][0], t.mat_[1][1], t.mat_[2][2],
            t.mat_[0][1], t.mat_[0][2], t.mat_[1][2]);

    ++itr;
  }

  // Write tail, bottom part.


  fclose(f);
  return true;
}


static bool
write_elc_file(ProgressReporter *pr, FieldHandle fld, const char *filename)
{
  PointCloudMesh *mesh = dynamic_cast<PointCloudMesh *>(fld->mesh().get_rep());
  if (mesh == 0)
  {
    pr->error("Field does not contain a PointCloudMesh.");
    return false;
  }

  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
    pr->error(string("Unable to open file '") + filename + "' for writing.");
    return false;
  }

  fclose(f);
  return true;
}



static bool
write_dip_file(ProgressReporter *pr, FieldHandle fld, const char *filename)
{
  PointCloudField<Vector> *pcv =
    dynamic_cast<PointCloudField<Vector> *>(fld.get_rep());
  if (pcv == 0)
  {
    pr->error("Field is not a vector point cloud.");
    return false;
  }

  FILE *f = fopen(filename, "w");
  if (f == NULL)
  {
    pr->error(string("Unable to open file '") + filename + "' for writing.");
    return false;
  }

  fclose(f);
  return true;
}


bool
ForwardIPM::write_par_file(string filename)
{
  FILE *f = fopen(filename.c_str(), "w");
  if (f == NULL)
  {
    error("Unable to open file '" + filename + "' for writing.");
    return false;
  }

  fprintf(f, "Random parameters\n");
  
  fclose(f);
  return true;
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
  
  MatrixIPort *dipole_port = (MatrixIPort *)get_iport("Dipole Positions");
  MatrixHandle dipole;
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
  const string tmplog = tmpdir + "forward.log";
  const string resultfile = tmpdir + "result.msr";
  const string condmeshfile = tmpdir + "condmesh.geo";
  const string condtensfile = tmpdir + "ca_perm.knw";
  const string electrodefile = tmpdir + "electrode.elc";
  const string dipolefile = tmpdir + "dipole.dip";
  const string parafile = tmpdir + "forward.par";

  try {
    // Make our tmp directory.
    mode_t umsk = umask(00);
    if (mkdir(tmpdir.c_str(), 0700) == -1)
    {
      error("Unable to open a temporary working directory.");
      throw false;
    }
    umask(umsk);

    // Write out condmesh
    if (!write_geo_file(this, condmesh, condmeshfile.c_str()))
    {
      error("Unable to export CondMesh geo file.");
      throw false;
    }
    // Write out condmesh
    if (!write_knw_file(this, condmesh, condtensfile.c_str()))
    {
      error("Unable to export CondMesh knw file.");
      throw false;
    }


    // Write out Electrodes
    if (!write_elc_file(this, electrodes, electrodefile.c_str()))
    {
      error("Unable to export electrode file.");
      throw false;
    }

    // Write out Dipole
    //if (!write_dip_file(this, dipole, dipolefile.c_str()))
    {
      error("Unable to export dipole file.");
      throw false;
    }

    // Write out our parameter file.
    if (!write_par_file(parafile)) { throw false; }

    // Construct our command line.
    const string impfile = "imp";
    const string command = "(cd tmpfile;" + impfile + " -i sourcesimulation" +
      " -h " + condmeshfile + " -p " + parafile + " -s " + electrodefile +
      " -d " + dipolefile + " -o " + resultfile + " -fwd FEM -sens EEG)";

    // Execute the command.  Last arg just a tmp file for logging.
    if (!Exec_execute_command(this, command, tmplog))
    {
      error("The ipm program failed to run for some unknown reason.");
      throw false;
    }

    // Read in the results and send them along.


    throw true; // cleanup.
  }
  catch (...)
  {
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
}


} // End namespace SCIRun
