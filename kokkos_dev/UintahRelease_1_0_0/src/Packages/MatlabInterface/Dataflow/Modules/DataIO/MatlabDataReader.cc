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
 * FILE: MatlabDataReader.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 30 MAR 2004
 */
 
/* 
 * This module reads a matlab file and converts it to a SCIRun matrix
 *
 */


/* 
 * This file was adapted from mlMatricesReader.h 
 */

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabconverter.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Network/Ports/StringPort.h>

namespace MatlabIO {

using namespace SCIRun;

class MatlabDataReader : public Module 
{

  public:

    // Constructor
    MatlabDataReader(GuiContext*);

    // Destructor
    virtual ~MatlabDataReader();

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
    
    void		indexmatlabfile(bool postmsg);
    
    // readmatlabarray():
    //   This function reads the in the gui selected matlab array. It
    //   retrieves the current filename and the matrix selected in this
    //   file and returns an object containing the full matrix
    
    matlabarray readmatlabarray(long p);

    // displayerror()
    //   Relay an error during the matrixselection process directly to
    //   the user
    
    void displayerror(std::string str);

  private:

    enum { NUMPORTS = 9};
    
    // GUI variables
    GuiString				guifilename_;		// .mat filename (import from GUI)
    GuiString       guifilenameset_;
    GuiString				guimatrixinfotextslist_;   	// A list of matrix-information strings of the contents of a .mat-file
    GuiString				guimatrixnameslist_;	// A list of matrix-names of the contents of a .mat-file 
    GuiString				guimatrixname_;		// the name of the matrix that has been selected

    // Ports (We only use one output port)
    SCIRun::FieldOPort*			ofield_[3];
    SCIRun::MatrixOPort*			omatrix_[3];
    SCIRun::NrrdOPort*			onrrd_[3];
};

DECLARE_MAKER(MatlabDataReader)

// Constructor:
// Initialise all the variables shared between TCL and SCIRun
// Only filename and matrixname are used to reconstruct the 
// settings of a previously created module
// matrixinfotexts and matrixnames serve as outputs to TCL.

MatlabDataReader::MatlabDataReader(GuiContext* ctx)
  : Module("MatlabDataReader", ctx, Source, "DataIO", "MatlabInterface"),
    guifilename_(get_ctx()->subVar("filename")),
    guifilenameset_(get_ctx()->subVar("filename-set")),
    guimatrixinfotextslist_(get_ctx()->subVar("matrixinfotextslist")),     
    guimatrixnameslist_(get_ctx()->subVar("matrixnameslist")),    
    guimatrixname_(get_ctx()->subVar("matrixname"))
{
  indexmatlabfile(false);
}

// Destructor:
// All my objects have descrutors and hence nothing needs
// explicit descruction
MatlabDataReader::~MatlabDataReader()
{
}

// Execute:
// Inner workings of this module
void MatlabDataReader::execute()
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
  
  // If the filename is empty, launch an error
  if (filename == "")
  {
    error("MatlabDataReader: No file name was specified");
    return;
  }

  try
  {
    for (long p=0;p<3;p++)
    {

      // Find the output port the scheduler has created 
      ofield_[p] = static_cast<SCIRun::FieldOPort *>(get_oport(static_cast<int>(p)));

      if(!ofield_[p]) 
      {
        error("MatlabDataReader: Unable to initialize output field port");
        return;
      }

      matlabarray ma = readmatlabarray(p);
      if (ma.isempty())
      {
        continue;
      }

      SCIRun::FieldHandle mh;

      matlabconverter translate(dynamic_cast<SCIRun::ProgressReporter*>(this));
      translate.mlArrayTOsciField(ma,mh);
      ofield_[p]->send(mh);
    }

    for (long p=0;p<3;p++)
    {

      // Find the output port the scheduler has created 
      omatrix_[p] = static_cast<SCIRun::MatrixOPort *>(get_oport(static_cast<int>(p)+3));

      if(!omatrix_[p]) 
      {
        error("MatlabDataReader: Unable to initialize output matrix port");
        return;
      }

      matlabarray ma = readmatlabarray(p+3);
      if (ma.isempty())
      {
        continue;
      }

      SCIRun::MatrixHandle mh;
      matlabconverter translate(dynamic_cast<SCIRun::ProgressReporter*>(this));
      translate.mlArrayTOsciMatrix(ma,mh);
      omatrix_[p]->send(mh);
    }

    for (long p=0;p<3;p++)
    {

      // Find the output port the scheduler has created 
      onrrd_[p] = static_cast<SCIRun::NrrdOPort *>(get_oport(static_cast<int>(p)+6));

      if(!onrrd_[p]) 
      {
        error("MatlabDataReader: Unable to initialize output nrrd port");
        return;
      }

      matlabarray ma = readmatlabarray(p+6);
      if (ma.isempty())
      {
        continue;
      }

      SCIRun::NrrdDataHandle mh;
      matlabconverter translate(dynamic_cast<SCIRun::ProgressReporter*>(this));
      translate.mlArrayTOsciNrrdData(ma,mh);
      onrrd_[p]->send(mh);
    }
    
    SCIRun::StringHandle filenameH = scinew String(filename);
    send_output_handle("Filename",filenameH,true);    
  }

  // in case something went wrong
          
  catch (matlabfile::could_not_open_file)
  {
    error("MatlabDataReader: Could not open file");
  }
  catch (matlabfile::invalid_file_format)
  {
    error("MatlabDataReader: Invalid file format");
  }
  catch (matlabfile::io_error)
  {
    error("MatlabDataReader: IO error");
  }
  catch (matlabfile::out_of_range)
  {
    error("MatlabDataReader: Out of range");
  }
  catch (matlabfile::invalid_file_access)
  {
    error("MatlabDataReader: Invalid file access");
  }
  catch (matlabfile::empty_matlabarray)
  {
    error("MatlabDataReader: Empty matlab array");
  }
  catch (matlabfile::matfileerror) 
  {
    error("MatlabDataReader: Internal error in reader");
  }
}


void MatlabDataReader::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("MatlabDataReader needs a minor command");
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


matlabarray MatlabDataReader::readmatlabarray(long p)
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



void MatlabDataReader::indexmatlabfile(bool postmsg)
{
	
  std::string filename = "";
  std::string matrixinfotexts[NUMPORTS];
  std::string matrixnames[NUMPORTS];
  std::string matrixinfotextslist = "";
  std::string matrixnameslist = "";
  std::string newmatrixname = "";
  std::string matrixname = "";
  
  guimatrixinfotextslist_.set(matrixinfotextslist);
  guimatrixnameslist_.set(matrixnameslist);
  
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
  
  for (long p=0;p<NUMPORTS;p++)
  {
    matrixinfotexts[p] = "{ ";
    matrixnames[p] = "{ ";
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
    long cindex = 0;		// compability index, which matlab array fits the SCIRun matrix best? 
    long maxindex = 0;		// highest index found so far
            
    // Scan the file and see which matrices are compatible
    // Only those will be shown (you cannot select incompatible matrices).
            
    std::string infotext;
    
    for (long p=0;p<mfile.getnummatlabarrays();p++)
    {
      ma = mfile.getmatlabarrayinfo(p); // do not load all the data fields
      for (long q=0;q<NUMPORTS;q++)
      {
        if ((q==0)||(q==1)||(q==2)) cindex = translate.sciFieldCompatible(ma,infotext);
        if ((q==3)||(q==4)||(q==5)) cindex = translate.sciMatrixCompatible(ma,infotext);
        if ((q==6)||(q==7)||(q==8)) cindex = translate.sciNrrdDataCompatible(ma,infotext);
        
        if (cindex)
        {
          // in case we need to propose a matrix to load, select
          // the one that is most compatible with the data
          if (cindex > maxindex) { maxindex = cindex; newmatrixname = ma.getname();}
  
          // create tcl style list to use in the array selection process
  
          matrixinfotexts[q] += std::string("{" + infotext + "} ");
          matrixnames[q] += std::string("{" + ma.getname() + "} "); 
          if (ma.getname() == matrixnamelist[q]) foundmatrixname[q] = true;
        }
      }
    }
    
    
    for (long q=0;q<NUMPORTS;q++)
    {
      matrixinfotexts[q] += "{none} } ";
      matrixnames[q] += "{<none>} } ";
    }
    
    mfile.close();

    // automatically select a matrix if the current matrix name
    // cannot be found or if no matrixname has been specified
    
    matrixname = "";
    for (long p=0;p<NUMPORTS;p++)
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
    
    for (long q=0;q<NUMPORTS;q++)
    {
      matrixinfotextslist += matrixinfotexts[q];
      matrixnameslist += matrixnames[q];
    }
    
    
    // Update TCL on the contents of this matrix
    guimatrixname_.set(matrixname);
    guimatrixinfotextslist_.set(matrixinfotextslist);
    guimatrixnameslist_.set(matrixnameslist);
  }
  
  // in case something went wrong
  // close the file and then dermine the problem

  catch (matlabfile::could_not_open_file)
  {
    displayerror("MatlabDataReader: Could not open file");
  }
  catch (matlabfile::invalid_file_format)
  {
    displayerror("MatlabDataReader: Invalid file format");
  }
  catch (matlabfile::io_error)
  {
    displayerror("MatlabDataReader: IO error");
  }
  catch (matlabfile::out_of_range)
  {
    displayerror("MatlabDataReader: Out of range");
  }
  catch (matlabfile::invalid_file_access)
  {
    displayerror("MatlabDataReader: Invalid file access");
  }
  catch (matlabfile::empty_matlabarray)
  {
    displayerror("MatlabDataReader: Empty matlab array");
  }
  catch (matlabfile::matfileerror) 
  {
    displayerror("MatlabDataReader: Internal error in reader");
  }
  return;
}


void MatlabDataReader::displayerror(std::string str)
{
  get_gui()->lock();
  // Explicit call to TCL
  get_gui()->execute("tk_messageBox -icon error -type ok -title {ERROR} -message {" + str + "}");
  get_gui()->unlock();
}


} // End namespace MatlabIO
