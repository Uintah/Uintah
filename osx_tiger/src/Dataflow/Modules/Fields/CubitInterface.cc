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
 *  CubitInterface.cc:  
 *
 *  Written by:
 *   McKay Davis
 *   Scientific Computing and Imaging INstitute
 *   University of Utah
 *   May 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/ImportExport/ExecConverter.h>
#include <string>
#include <vector>
#include <map>

namespace SCIRun {

class CubitInterface : public Module
{
private:
  FieldIPort *		iport1_;
  FieldIPort *		iport2_;
  FieldOPort *		oport_;
  int			field1_generation_;
  int			pcf_generation_;
  GuiString		cubitdir_;
  

  bool			write_facet_file(string, TriSurfMeshHandle &);
  bool			write_journal_file(string, string, string);
public:
  CubitInterface(GuiContext* ctx);
  virtual ~CubitInterface();

  virtual void execute();
};


DECLARE_MAKER(CubitInterface)

CubitInterface::CubitInterface(GuiContext* ctx)
  : Module("CubitInterface", ctx, Filter, "FieldsCreate", "SCIRun"),
    field1_generation_(-1),
    pcf_generation_(-1),
    cubitdir_(ctx->subVar("cubitdir"))
{
}


CubitInterface::~CubitInterface()
{
}

bool
CubitInterface::write_facet_file(string filename, TriSurfMeshHandle &tsm)
{
  FILE *f = fopen(filename.c_str(), "wt");
  if (f == NULL)
  {
    error("Unable to open file"+filename+" for writing.");
    return false;
  }
  TriSurfMesh::Node::size_type tsmns;
  TriSurfMesh::Face::size_type tsmfs;
  TriSurfMesh::Node::iterator tsmn, tsmne;
  TriSurfMesh::Face::iterator tsmf, tsmfe;
  TriSurfMesh::Node::array_type face_nodes;
  tsm->size(tsmns);
  tsm->size(tsmfs);
  tsm->begin(tsmn);
  tsm->end(tsmne);
  tsm->begin(tsmf);
  tsm->end(tsmfe);

  fprintf (f,"%d %d\n", int(tsmns), int(tsmfs));
  Point p;
  while (tsmn != tsmne) {
    tsm->get_point(p, *tsmn);
    fprintf (f, "%d %f %f %f\n", (*tsmn).index_, p.x(), p.y(), p.z());
    ++tsmn;
  }

  while (tsmf != tsmfe) {
    tsm->get_nodes(face_nodes, *tsmf);
    fprintf (f, "%d %d %d %d\n", (*tsmf).index_, 
	     face_nodes[0].index_, face_nodes[1].index_, face_nodes[2].index_);
    ++tsmf;
  }

  fclose(f);
  remark("Wrote: "+filename);
  return true;
}
    

bool
CubitInterface::write_journal_file(string filename, string facetfilename,
				   string outputfilename)
{
  FILE *f = fopen(filename.c_str(), "wt");
  if (f == NULL)
  {
    error("Unable to open file "+filename+" for writing.");
    return false;
  }
  fprintf(f, "reset\n");
  fprintf(f, "import facet \'%s\' feature 0 make\n", facetfilename.c_str());
  fprintf(f, "volume 1 scheme tetmesh\n");
  fprintf(f, "volume 1 size 1\n");
  fprintf(f, "mesh volume 1\n");
  fprintf(f, "set large exodus file off\n");
  fprintf(f, "export mesh \'%s\' dimension 3\n", outputfilename.c_str());
  fclose(f);
  remark("Wrote: "+filename);
  return true;
}




void
CubitInterface::execute()
{
  iport1_ = (FieldIPort *)get_iport("Field");
  iport2_ = (FieldIPort *)get_iport("PointCloudField");
  oport_  = (FieldOPort *)get_oport("Field");


  FieldHandle field1;
  iport1_->get(field1);
  if (!field1.get_rep()) {
    error("No input field to port 1.");
    return;
  }

  // Get the PointCloudField from the seco port
  FieldHandle pcf;
  iport2_->get(pcf);


  if (field1->get_type_description()->get_name().find("TriSurfField")) {
    error("Field connected to port 1 must be TriSurfField.");
    return;
  }

  // Make sure the second input field is of type PointCloudField
  if (pcf.get_rep() && 
      pcf->get_type_description()->get_name().find("PointCloudField")) {
    error("Field connected to port 1 must be PointCloudField.");
    return;
  }

#if 0
  // Return if nothing in gui or fields has changed
  if (pcf->generation == pcf_generation_ &&
      field1->generation == field1_generation_) return;

  pcf_generation_ = pcf->generation;
  field1_generation_ = field1->generation;
#endif

  string tmp(sci_getenv("SCIRUN_TMP_DIR"));
  string facetfile = tmp+"/cubit.facet";
  string journalfile = tmp+"/cubit.journal";
  string outputfile = tmp+"/cubit.exo";
  string logfile = tmp+"/cubit.log";

  TriSurfMeshHandle tsm = (TriSurfMesh *)(field1->mesh().get_rep());
  if (!write_facet_file(facetfile, tsm)) return;
  if (!write_journal_file(journalfile, facetfile, outputfile)) return;
  
  string cubitdir = cubitdir_.get();//"/scratch/claro";
  string command = "export LD_LIBRARY_PATH="+cubitdir+"/libs:"+cubitdir+"/bin;"+
    cubitdir+"/bin/clarox "+journalfile;

  msgStream_flush();
  if (!Exec_execute_command(this, command, logfile))
  {
    error("The program failed to run for some unknown reason.");
    throw false;
  }
  msgStream_flush();



  oport_->send(field1);
}


}
