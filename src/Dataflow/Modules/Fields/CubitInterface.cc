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
#include <Core/Containers/StringUtil.h>
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
#include <iostream>
#include <fstream>
#include <sstream>


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
  GuiString		ncdump_;
  string		base_;
  
  void			cleanup(string error="");
  bool			write_facet_file(string, TriSurfMeshHandle &);
  bool			read_netcdf_file(string, TetVolMeshHandle &);
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
    cubitdir_(ctx->subVar("cubitdir")),
    ncdump_(ctx->subVar("ncdump"))
{
}


CubitInterface::~CubitInterface()
{
}


void
CubitInterface::cleanup(string error) {
  unlink((base_+"exodus").c_str());
  unlink((base_+"facet").c_str());
  unlink((base_+"journal").c_str());
  unlink((base_+"netcdf").c_str());
  unlink((base_+"log").c_str());
  if (!error.empty())
    throw error;
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

  TriSurfMesh::Edge::array_type tsmea;
  TriSurfMesh::Face::index_type neigh;
  
  tsm->synchronize(Mesh::EDGES_E | Mesh:: EDGE_NEIGHBORS_E);
		   
  while (tsmf != tsmfe) {
    tsm->get_edges(tsmea, *tsmf);
    for (int e = 0; e < tsmea.size(); ++e) {
      if (!tsm->get_neighbor(neigh, *tsmf, tsmea[e])) {
	cleanup("Edge#"+to_string(tsmea[e].index_)+" is a boundary face.\n"+
		"The surface is not closed and cannot be meshed.");
      }
    }
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
  fprintf(f, "exit\n");
  fclose(f);
  remark("Wrote: "+filename);
  return true;
}

struct FillNodeFtor {
  Point operator()(Point &p) {
    return p;
  }
};

struct FillCellFtor {
  FillCellFtor(vector<int> &array) : array_(array) {}
  vector <int> array_;
  int ret_val[4];
  int *operator()(int n) {
    for (int i = 0; i < 4; ++i)
      ret_val[i] = array_[n*4+i];
    return ret_val;
  }
};

struct SimpleIter {
  SimpleIter(int init = 0) : i(init) {}
  SimpleIter & operator++() { ++i; return *this; };
  bool operator!=(const SimpleIter &right) { return i != right.i; };
  int operator-(const SimpleIter &right) { return i - right.i; };
  int operator*() { return i; };
  int i;
};


bool
CubitInterface::read_netcdf_file(string filename, TetVolMeshHandle &mesh)
{
  ifstream infile;
  infile.open(filename.c_str());
  infile.sync();
  infile.seekg(0,ios_base::beg);
  int nnodes = -1;
  int nelems = -1;
  string instring;

  remark("Reading NetCDF file: "+filename);
  while (!infile.eof()) {
    // Eat up spurious blank lines.
    while (infile.peek() == ' ' ||
	   infile.peek() == '\t' ||
	   infile.peek() == '\n')
    {
      if (infile.bad()) return false;
      infile.get();
      if (infile.bad()) return false;
    }

    // Eat lines that start with #, they are comments.
    if (infile.peek() == '#')
    {
      if (infile.bad()) return false;
      infile.ignore(1024,'\n');
      if (infile.bad()) return false;
      continue;
    }

    if (infile.bad()) return false;

    infile >> instring;
    if (infile.bad()) return false;
    if (instring == "num_nodes") {
      infile >> instring >> nnodes;
      if (infile.bad()) return false;
    } else if (instring == "num_elem") {
      infile >> instring >> nelems;
      if (infile.bad()) return false;
    }
    
    infile.ignore(1024,'\n');
    if (nnodes != -1 && nelems != -1) break;
  }

  if (nnodes == -1 || nelems == -1) return false;

  instring = "";
  while (infile.good() && instring != "data:") {
    infile >> instring;
    if (infile.bad()) return false;
    infile.ignore(1024,'\n');
    if (infile.bad()) return false;
  }

  int e, i;
  double d;
  vector <int> elems;
  vector <Point> nodes;
  elems.resize(nelems*4);
  nodes.resize(nnodes);
  remark("Number of nodes in mesh: "+to_string(nnodes));
  remark("Number of elements in mesh: "+to_string(nelems));
  while (!infile.eof()) {
    infile >> instring;
    if (infile.bad()) return false;
    infile.ignore(1024,'\n');    
    if (infile.bad()) return false;
    if (instring == "connect1") {
      remark("Trying to read point connections.");
      for (i = 0; i < nelems*4; ++i) {
	infile >> instring;
	if (infile.bad()) return false;
	if (i == nelems*4-1) instring = instring+",";
	if (!string_to_int(instring.substr(0, instring.size()-1),e)) {
	  error ("Failed to convert"+instring);
	} else {
	  elems[i] = e-1;
	}
      }
      remark("Successfully read connecitons.");
    }

    if (instring == "coord") {
      remark("Trying to read point coordinates.");
      for (e = 0; e < 3; ++e) {
	for (i = 0; i < nnodes; ++i) {
	  infile >> instring;
	  if (infile.bad()) return false;
	  if (e == 2 && i == nnodes-1) instring = instring+",";
	  if (!string_to_double(instring.substr(0,instring.size()-1),d)) {
	    error ("Failed to convert "+instring);
	  } else {
	    nodes[i](e) = d;
	  }
	}
      }
      remark("Successfully read coordinates.");
    }
  }
  
  TetVolMesh *tvm = scinew TetVolMesh();
  SimpleIter beg(0), end(nelems);
  tvm->fill_points(nodes.begin(), nodes.end(), FillNodeFtor());
  tvm->fill_cells(beg, end, FillCellFtor(elems));

  mesh = tvm;

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
  
  string tmp(sci_getenv("SCIRUN_TMP_DIR")+string("/"));
  //  string rand(to_string(int(drand48()*1000.00)));
  base_ = tmp+id+"."+string(sci_getenv("USER"))+".";//+rand+".";

  string facetfile = base_+"facet";
  string journalfile = base_+"journal";
  string exodusfile = base_+"exodus";
  string netcdffile = base_+"netcdf";
  string logfile = base_+"log";

  TriSurfMeshHandle tsm = (TriSurfMesh *)(field1->mesh().get_rep());
  if (!write_facet_file(facetfile, tsm)) {
    cleanup("Cannot write facet file: "+facetfile);
  }

  if (!write_journal_file(journalfile, facetfile, exodusfile)) {
    cleanup("Cannot write journal file: "+journalfile);
  }
  
  string command = 
    "export LD_LIBRARY_PATH="+cubitdir_.get()+"/libs:"+cubitdir_.get()+"/bin;"+
    cubitdir_.get()+"/bin/clarox "+journalfile+";";
  
  if (!Exec_execute_command(this, command, logfile))
  {
    cleanup("The Cubit program failed to run for some unknown reason.");
  }

  command = 
    ncdump_.get()+" -v connect1,coord "+exodusfile+" > "+netcdffile+
    "; echo Done converting "+exodusfile;

  if (!Exec_execute_command(this, command, netcdffile))
  {
    cleanup("The ncdump program failed to run for some unknown reason.");
  }

  TetVolMeshHandle mesh;
  if (!read_netcdf_file(netcdffile, mesh)) {
    cleanup("Error read NetCDF file: "+netcdffile);
  }

  cleanup();

  TetVolField<double> *f = scinew TetVolField<double>(mesh, 0);
  oport_->send(f);
}


}
