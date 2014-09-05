/*
 *  Mesh3dReader.cc:
 *
 *  Written by:
 *   veselin
 *   TODAY'S DATE HERE
 *
 */

#include <fstream>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <sys/stat.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>

#include <Dataflow/share/share.h>

namespace DDDAS {

using namespace SCIRun;
using namespace std;

class PSECORESHARE Mesh3dReader : public Module
{
   private:
      GuiString filename_;
      FieldHandle handle_;
      FieldHandle handle2_;
      MatrixHandle mhandle_;
      MatrixHandle mhandle2_;
      string    old_filename_;
      time_t    old_filemodification_;

   public:
      Mesh3dReader(GuiContext*);

      virtual ~Mesh3dReader();

      virtual void execute();

      virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(Mesh3dReader)
Mesh3dReader::Mesh3dReader(GuiContext* ctx)
  : Module("Mesh3dReader", ctx, Source, "DataIO", "DDDAS"),
    filename_(ctx->subVar("filename")),
    old_filemodification_(0)
{
   // empty
}

Mesh3dReader::~Mesh3dReader()
{
   // empty
}

void Mesh3dReader::execute()
{
  const string fn(filename_.get());

  // Read the status of this file so we can compare modification timestamps
  struct stat buf;
  if( fn == "" ) {
    error("No file has been selected.  Please choose a file.");
    return;
  } else if (stat(fn.c_str(), &buf)) {
    error("File '" + fn + "' not found.");
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif
  if (!handle_.get_rep() || 
      fn != old_filename_ || 
      new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_ = fn;
    ifstream mesh_stream(fn.c_str());
    if (!mesh_stream)
    {
      error("Error reading file '" + fn + "'.");
      return;
    }
    
    // Read the file
    {
       int i, npts, ntets, ntris;

       mesh_stream.ignore(256, '\n');  // ignore the mesh-type line
       mesh_stream >> npts;            // read the number of vertices (points)
       TetVolMesh *tvm = new TetVolMesh();
       TriSurfMesh *tsm = new TriSurfMesh();
       for (i = 0; i < npts; i++)      // read the coords of all the points
       {
          double x, y, z;
          mesh_stream >> x >> y >> z;
          tvm->add_point(Point(x,y,z));
          tsm->add_point(Point(x,y,z));
       }
       mesh_stream >> ntets;           // read the number of tets
       ColumnMatrix *cm = scinew ColumnMatrix(ntets);
       for (i = 0; i < ntets; i++)
       {
          int attr, v[4];
          mesh_stream >> attr >> v[0] >> v[1] >> v[2] >> v[3];
          tvm->add_tet(v[0]-1, v[1]-1, v[2]-1, v[3]-1);
          (*cm)[i] = attr;
       }
       mesh_stream >> ntris;
       ColumnMatrix *cm2 = scinew ColumnMatrix(ntris);
       for (i = 0; i < ntris; i++)
       {
          int attr, v[3];
          mesh_stream >> attr >> v[0] >> v[1] >> v[2];
          tsm->add_triangle(v[0]-1, v[1]-1, v[2]-1);
          (*cm2)[i] = attr;
       }

       TetVolField<double> *tv = scinew TetVolField<double>(tvm, -1);
       handle_  = (Field *)tv;
       TriSurfField<double> *ts = scinew TriSurfField<double>(tsm, -1);
       handle2_ = (Field *)ts;
       mhandle_ = (Matrix *)cm;
       mhandle2_ = (Matrix *)cm2;
       // do we need to delete something ?
    }

  }

  // Send the data downstream.
  SimpleOPort<FieldHandle> *outport = (SimpleOPort<FieldHandle> *)getOPort(0);
  if (!outport) {
    error("Unable to initialize oport 0.");
    return;
  }
  outport->send(handle_);

  outport = (SimpleOPort<FieldHandle> *)getOPort(1);
  if (!outport) {
    error("Unable to initialize oport 1.");
    return;
  }
  outport->send(handle2_);

  SimpleOPort<MatrixHandle> *oport2 = (SimpleOPort<MatrixHandle> *)getOPort(2);
  if (!oport2) {
    error("Unable to initialize oport 2.");
    return;
  }
  oport2->send(mhandle_);

  oport2 = (SimpleOPort<MatrixHandle> *)getOPort(3);
  if (!oport2) {
    error("Unable to initialize oport 3.");
    return;
  }
  oport2->send(mhandle2_);
}

void
 Mesh3dReader::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace DDDAS


