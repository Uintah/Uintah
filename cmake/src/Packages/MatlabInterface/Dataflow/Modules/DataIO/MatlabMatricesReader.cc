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
 * FILE: MatlabMatricesReader.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 30 MAR 2004
 */
 
/* 
 * This module reads a matlab file and converts it to a SCIRun matrix
 *
 */

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabconverter.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/StringPort.h>

namespace MatlabIO {

using namespace SCIRun;

class MatlabMatricesReader : public Module 
{

  public:

    // Constructor
    MatlabMatricesReader(GuiContext*);

    // Destructor
    virtual ~MatlabMatricesReader();

    // Std functions for each module
    // execute():
    //   Execute the module and put data on the output port
    //
    // tcl_command():
    //   Handles call-backs from the TCL code
    
    virtual void execute();
    virtual void tcl_command(GuiArgs&, void*);

  private:
    
    // indexmatlabfile():
    //   This functions is used in the GUI interface, it loads the
    //   currently selected .mat file and returns an information
    //   string to TCL with all the names and formats of the matrices.
    //   NOTE: This Function explicitly depends on the TCL code.
    
    void  indexmatlabfile(bool postmsg);
    
    // readmatlabarray():
    //   This function reads the in the gui selected matlab array. It
    //   retrieves the current filename and the matrix selected in this
    //   file and returns an object containing the full matrix
    
    matlabarray readmatlabarray(int p);

    // displayerror()
    //   Relay an error during the matrixselection process directly to
    //   the user
    
    void displayerror(std::string str);

  private:

    enum { NUMPORTS = 6};
    
    // GUI variables
    GuiString guifilename_; // .mat filename (import from GUI)
    GuiString guifilenameset_;
    GuiString guimatrixinfotexts_;  // A list of matrix-information strings of the contents of a .mat-file
    GuiString guimatrixnames_; // A list of matrix-names of the contents of a .mat-file 
    GuiString guimatrixname_; // the name of the matrix that has been selected
    GuiInt guidisabletranspose_; // Do not convert from Fortran ordering to C++ ordering
    
    // Ports (We only use one output port)
    MatrixOPort* omatrix_[NUMPORTS];
    
};

DECLARE_MAKER(MatlabMatricesReader)

// Constructor:
// Initialise all the variables shared between TCL and SCIRun
// Only filename and matrixname are used to reconstruct the 
// settings of a previously created module
// matrixinfotexts and matrixnames serve as outputs to TCL.

MatlabMatricesReader::MatlabMatricesReader(GuiContext* ctx)
  : Module("MatlabMatricesReader", ctx, Source, "DataIO", "MatlabInterface"),
    guifilename_(get_ctx()->subVar("filename")),
    guifilenameset_(get_ctx()->subVar("filename-set")),
    guimatrixinfotexts_(get_ctx()->subVar("matrixinfotexts")),     
    guimatrixnames_(get_ctx()->subVar("matrixnames")),    
    guimatrixname_(get_ctx()->subVar("matrixname")),
    guidisabletranspose_(get_ctx()->subVar("disable-transpose"))
{
	indexmatlabfile(false);
}

// Destructor:
// All my objects have descrutors and hence nothing needs
// explicit descruction
MatlabMatricesReader::~MatlabMatricesReader()
{
}

// Execute:
// Inner workings of this module
void MatlabMatricesReader::execute()
{

  StringIPort *filenameport;
  if ((filenameport = static_cast<StringIPort *>(get_input_port("Filename"))))
  {
    StringHandle stringH;
    if (filenameport->get(stringH))
    {
      if (stringH.get_rep())
      {
        std::string filename = stringH->get();
        guifilename_.set(filename);
        get_ctx()->reset();
      }
    }
  }

  // Get the filename from TCL.
  std::string filename = guifilename_.get();
  int disable_transpose = guidisabletranspose_.get();
  matlabconverter translate(dynamic_cast<SCIRun::ProgressReporter *>(this));
  translate.setdisabletranspose(disable_transpose);
  
  // If the filename is empty, launch an error
  if (filename == "")
  {
    error("MatlabMatricesReader: No file name was specified");
    return;
  }

  try
  {
    for (int p=0;p<NUMPORTS;p++)
    {

      // Find the output port the scheduler has created 
      omatrix_[p] = static_cast<MatrixOPort *>(get_oport(static_cast<int>(p)));

      if(!omatrix_[p]) 
      {
        error("MatlabMatricesReader: Unable to initialize output port");
        return;
      }

      // Now read the matrix from file
      // The next function will open, read, and close the file
      // Any error will be exported as an exception.
      // The matlab classes are all based in the matfilebase class
      // which carries the definitions of the exceptions. These
      // definitions are inherited by all other "matlab classes"
      
      matlabarray ma = readmatlabarray(p);

      // An empty array means something must have gone wrong
      // Or there is no data to put on this port.
      // Do not translate empty arrays, but continue to the 
      // next output port.

      if (ma.isempty())
      {
              continue;
      }

      // The data is still in matlab format and the next function
      // creates a SCIRun matrix object

      SCIRun::MatrixHandle mh;
      translate.mlArrayTOsciMatrix(ma,mh);
      
      // Put the SCIRun matrix in the hands of the scheduler
      omatrix_[p]->send(mh);
    }
    
    SCIRun::StringHandle filenameH = scinew String(filename);
    send_output_handle("Filename",filenameH,true);    
  }

  // in case something went wrong
    
  catch (matlabfile::could_not_open_file)
  {
    error("MatlabMatricesReader: Could not open file");
  }
  catch (matlabfile::invalid_file_format)
  {
    error("MatlabMatricesReader: Invalid file format");
  }
  catch (matlabfile::io_error)
  {
    error("MatlabMatricesReader: IO error");
  }
  catch (matlabfile::out_of_range)
  {
    error("MatlabMatricesReader: Out of range");
  }
  catch (matlabfile::invalid_file_access)
  {
    error("MatlabMatricesReader: Invalid file access");
  }
  catch (matlabfile::empty_matlabarray)
  {
    error("MatlabMatricesReader: Empty matlab array");
  }
  catch (matlabfile::matfileerror) 
  {
    error("MatlabMatricesReader: Internal error in reader");
  }
}


void MatlabMatricesReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2){
    args.error("MatlabMatricesReader needs a minor command");
    return;
  }

  if( args[1] == "indexmatlabfile" )
  {
    
    // It turns out that in the current design, SCIRun reads variables once
    // and then assumes they do not change and hence caches the data
    // Why it is done so is unclear to me, but in order to have interactive
    // GUIs I need to reset the context. (this synchronises the data again)
    get_ctx()->reset();
    
    // Find out what the .mat file contains
    indexmatlabfile(true);
    return;
  }
  else 
  {
    // Relay data to the Module class
    Module::tcl_command(args, userdata);
  }
}


matlabarray MatlabMatricesReader::readmatlabarray(int p)
{
  matlabarray marray;
  std::string filename = guifilename_.get();
  std::string guimatrixname = guimatrixname_.get();
  std::string matrixname = "";

  // guimatrixname is a list with the name of the matrices per port
  // use the TCL command lindex to select the proper string from the list

  std::ostringstream oss;
  oss << "lindex {" << guimatrixname << "} " << p;

  get_gui()->lock();
  get_gui()->eval(oss.str(),matrixname);
  get_gui()->unlock();

  if (matrixname == "")
  {
    // return an empty array
    return(marray);
  }

  if (matrixname == "<none>")
  {
    // return an empty array
    return(marray);
  }

  // this block contains the file IO
  // The change of errors is reasonable
  // hence errors are generated as exceptions
    
  // having a local matfile object here ensures
  // the file will be closed (destructor of the object).

  matlabfile  mfile;
  mfile.open(filename,"r");
  marray = mfile.getmatlabarray(matrixname);
  mfile.close();

  return(marray);
}



void MatlabMatricesReader::indexmatlabfile(bool postmsg)
{

  std::string filename = "";
  std::string matrixinfotexts = "";
  std::string matrixnames = "";
  std::string newmatrixname = "";
  std::string matrixname = "";

  guimatrixinfotexts_.set(matrixinfotexts);
  guimatrixnames_.set(matrixnames);

  SCIRun::ProgressReporter* pr = 0;
  if (postmsg) pr = dynamic_cast<SCIRun::ProgressReporter* >(this);
  matlabconverter translate(pr);
  
  filename = guifilename_.get();	

  if (filename == "") 
  {
    // No file has been loaded, so reset the
    // matrix name variable
    guimatrixname_.set(newmatrixname);
    return;
  }

  matrixname = guimatrixname_.get();

  std::vector<std::string> matrixnamelist(NUMPORTS);
  bool foundmatrixname[NUMPORTS];

  for (int p=0;p<NUMPORTS;p++)
  {
    // TCL Dependent code
    std::ostringstream oss;
    oss << "lindex { " << matrixname << " } " << p;
    get_gui()->lock();
    get_gui()->eval(oss.str(),matrixnamelist[p]);
    get_gui()->unlock();
    foundmatrixname[p] = false;
  }

  try
  {
    matlabfile mfile;
    // Open the .mat file
    // This function also scans through the file and makes
    // sure it is amat file and counts the number of arrays
    
    mfile.open(filename,"r");
    
    // all matlab data is stored in a matlabarray object
    matlabarray ma;
    int cindex = 0;		// compability index, which matlab array fits the SCIRun matrix best? 
    int maxindex = 0;		// highest index found so far
            
    // Scan the file and see which matrices are compatible
    // Only those will be shown (you cannot select incompatible matrices).
            
    std::string infotext;
    
    for (int p=0;p<mfile.getnummatlabarrays();p++)
    {
      ma = mfile.getmatlabarrayinfo(p); // do not load all the data fields
      if ((cindex = translate.sciMatrixCompatible(ma,infotext)))
      {
        // in case we need to propose a matrix to load, select
        // the one that is most compatible with the data
        if (cindex > maxindex) { maxindex = cindex; newmatrixname = ma.getname();}
        
        // create tcl style list to use in the array selection process
        
        matrixinfotexts += std::string("{" + infotext + "} ");
        matrixnames += std::string("{" + ma.getname() + "} "); 
        for (int q=0;q<NUMPORTS;q++)
        {
          if (ma.getname() == matrixnamelist[q]) foundmatrixname[q] = true;
        }
      }
    }

    matrixinfotexts += "{none} ";
    matrixnames += "{<none>} ";

    mfile.close();

    // automatically select a matrix if the current matrix name
    // cannot be found or if no matrixname has been specified
    
    matrixname = "";
    for (int p=0;p<NUMPORTS;p++)
    {
      if (foundmatrixname[p] == false) 
      {   
        if (p==0) 
        {
          matrixnamelist[p] = newmatrixname;
        }
        else
        {
          matrixnamelist[p] = "<none>";
        }
      }
      matrixname += "{" + matrixnamelist[p] + "} ";
    }
    
    // Update TCL on the contents of this matrix
    guimatrixname_.set(matrixname);
    guimatrixinfotexts_.set(matrixinfotexts);
    guimatrixnames_.set(matrixnames);
  }

  // in case something went wrong
  // close the file and then dermine the problem

  catch (matlabfile::could_not_open_file)
  {
    displayerror("MatlabMatricesReader: Could not open file");
  }
  catch (matlabfile::invalid_file_format)
  {
    displayerror("MatlabMatricesReader: Invalid file format");
  }
  catch (matlabfile::io_error)
  {
    displayerror("MatlabMatricesReader: IO error");
  }
  catch (matlabfile::out_of_range)
  {
    displayerror("MatlabMatricesReader: Out of range");
  }
  catch (matlabfile::invalid_file_access)
  {
    displayerror("MatlabMatricesReader: Invalid file access");
  }
  catch (matlabfile::empty_matlabarray)
  {
    displayerror("MatlabMatricesReader: Empty matlab array");
  }
  catch (matlabfile::matfileerror) 
  {
    displayerror("MatlabMatricesReader: Internal error in reader");
  }
  return;
}


void MatlabMatricesReader::displayerror(std::string str)
{
  get_gui()->lock();
  // Explicit call to TCL
  get_gui()->execute("tk_messageBox -icon error -type ok -title {ERROR} -message {" + str + "}");
  get_gui()->unlock();
}


} // End namespace MatlabIO
