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
 * FILE: MatlabMatricesWriter.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 30 MAR 2004
 */ 

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/StringPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabconverter.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/String.h>

namespace MatlabIO {

using namespace SCIRun;

class MatlabMatricesWriter : public Module 
{

  public:

    // Constructor
    MatlabMatricesWriter(GuiContext*);

    // Destructor
    virtual ~MatlabMatricesWriter();

    // Std functions for each module
    // execute():
    //   Execute the module and put data on the output port
    
    virtual void execute();
    
  private:

    // functions for converting SCIRun matrices into matlab matrices
    // This class contains a scope of functions for translating data
    // structures.
    matlabconverter translate_;

    // Support functions for converting between TCL and C++
    // converttcllist:
    // TCL lists are stings in which the elements are separated by {}.
    // In a C++ environment it is easier to have them as a STL vector
    // object. This function converts the TCL list into an STL object
    // convertdataformat:
    // In TCL the dataformat is a string this function is used to convert
    // it into its matlab enum counterpart.
    std::vector<std::string> converttcllist(std::string str);
    matlabarray::mitype convertdataformat(std::string dataformat);

    // Support functions for the GUI
    // displayerror:
    // Directly reporting an error to the user (not in the error log)
    // overwrite:
    // Ask for confirmation to overwrite the file if it already exists
    
    void displayerror(std::string str);
    bool overwrite();

  private:

    enum { NUMPORTS = 6};
    
    // GUI variables
    GuiString guifilename_; // .mat filename (import from GUI)
    GuiString guifilenameset_;
    GuiString guimatrixname_; // A list of the matrix names
    GuiString guidataformat_; // A list of the dataformat for each matrix (int, double etc.)
    GuiString guimatrixformat_; // A list of the matlabarray format (numeric array, structured array)
  
};

DECLARE_MAKER(MatlabMatricesWriter)

// Constructor:
// Initialise all the variables shared between TCL and SCIRun
// matrixname contains a list of matrix names.
// dataformat contains a list of the format of each matrix (int32,single,double, etc...)
// matrixformat contains a list of the way the object is represented in matlab
// e.g. as a structured object or an object with the dataarray only

MatlabMatricesWriter::MatlabMatricesWriter(GuiContext* ctx)
  : Module("MatlabMatricesWriter", ctx, Sink, "DataIO", "MatlabInterface"),
    guifilename_(ctx->subVar("filename")),
    guimatrixname_(ctx->subVar("matrixname")),   
    guifilenameset_(ctx->subVar("filename-set")),
    guidataformat_(ctx->subVar("dataformat")),    
    guimatrixformat_(ctx->subVar("matrixformat"))
{
}

// Destructor:
// All my objects have descrutors and hence nothing needs
// explicit descruction
MatlabMatricesWriter::~MatlabMatricesWriter()
{
}

// Execute:
// Inner workings of this module

void MatlabMatricesWriter::execute()
{

  StringIPort *filenameport;
  if ((filenameport = static_cast<StringIPort *>(getIPort("filename"))))
  {
    StringHandle stringH;
    if (filenameport->get(stringH))
    {
      std::string filename = stringH->get();
      guifilename_.set(filename);
      ctx->reset();
    }
  }

  bool porthasdata[NUMPORTS];
  SCIRun::MatrixHandle matrixhandle[NUMPORTS];

  // find and evaluate which ports have been used

  SCIRun::MatrixIPort *iport;	
  for (long p=0; p<NUMPORTS; p++)
  {
    iport = static_cast<SCIRun::MatrixIPort *>(getIPort(p));
    if (!iport) 
    {
      error("MatlabMatricesWriter: Unable to initialize iport");
      return;
    }
    
    if (!iport->get(matrixhandle[p]) || !matrixhandle[p].get_rep())
    {
      porthasdata[p] = false;
    }
    else
    {
      porthasdata[p] = true;
    }
  }

  // Reorder the TCL input and put them
  // in orderly STL style vectors.

  // First update the GUI to C++ interface
  gui->execute(id+" Synchronise");
  ctx->reset();

  // Get the contents of the filename entrybox
  std::string filename = guifilename_.get();

  // Make sure we have a .mat extension
  int filenamesize = filename.size();
  if (filenamesize < 4) 
  {   
    filename += ".mat";
  }
  else
  {
    if (filename.substr(filenamesize-4,filenamesize) != ".mat") filename += ".mat";
  }

  // If the filename is empty, launch an error
  if (filename == "")
  {
    error("MatlabMatricesWriter: No file name was specified");
    return;
  }

  if (!overwrite()) return;

  // get all the settings from the GUI
  
  std::vector<std::string> matrixname;
  std::vector<std::string> dataformat;
  std::vector<std::string> matrixformat;
  
  matrixname = converttcllist(guimatrixname_.get());
  dataformat = converttcllist(guidataformat_.get());
  matrixformat = converttcllist(guimatrixformat_.get());
  
  // Check the validity of the matrixnames

  for (long p=0;p<static_cast<long>(matrixname.size());p++)
  {
    if (porthasdata[p] == false) continue; // Do not check not used ports
    if (!translate_.isvalidmatrixname(matrixname[p]))
    {
      error("MatlabMatricesWriter: The matrix name specified is invalid");
      return;
    }
    for (long q=0;q<p;q++)
    {
      if (porthasdata[q] == false) continue;
      if (matrixname[q] == matrixname[p])
      {
        error("MatlabMatricesWriter: A matrix name is used twice");
        return;
      }
    }
  }
	
  try
  {
    matlabfile mfile;   // matlab file object contains all function for reading and writing matlab arrayd
    matlabarray ma;		// matlab style formatted array (can be stored in a matlabfile object)
    mfile.open(filename,"w");   // open file for writing
    
    // Add an information tag to the data, so the origin of the file is known
    // There are 116 bytes of free data storage at the header of the file.
    // Do not start the file with 'SCI ', otherwise the file looks like a
    // native SCIRun file which uses the same extension.
    
    mfile.setheadertext("Matlab V5 compatible file generated by SCIRun [module MatlabMatricesWriter version 1.1]");
    
    for (long p=0;p<NUMPORTS;p++)
    {
      if (porthasdata[p] == false) continue; // if the port is not connected skip to the next one

      // Convert the SCIRun matrixobject to a matlab object

      if (matrixformat[p] == "struct array")
      {   
        // translate the matrix into a matlab structured array, which
        // can also store some data from the property manager
        translate_.converttostructmatrix();
        translate_.setdatatype(convertdataformat(dataformat[p]));
      }

      if (matrixformat[p] == "numeric array")
      {
        // only store the numeric parts of the data
        translate_.converttonumericmatrix();
        translate_.setdatatype(convertdataformat(dataformat[p]));
      }

      translate_.sciMatrixTOmlArray(matrixhandle[p],ma,static_cast<SCIRun::Module *>(this));	
      if (ma.isempty())
      {
        warning("One of the matrices is empty");
        continue; // Do not write empty matrices
      }
      // Every thing seems OK, so proceed and store the matrix in the file
      mfile.putmatlabarray(ma,matrixname[p]);
    }
    
    mfile.close();
  }	

  // in case something went wrong
    
  catch (matlabfile::could_not_open_file)
  {
    error("MatlabMatricesWriter: Could not open file");
  }
  catch (matlabfile::invalid_file_format)
  {
    error("MatlabMatricesWriter: Invalid file format");
  }
  catch (matlabfile::io_error)
  {   // IO error from ferror
    error("MatlabMatricesWriter: IO error");
  }
  catch (matlabfile::matfileerror) 
  {   // All other errors are classified as internal
    // matfileerrror is the base class on which all
    // other exceptions are based.
    error("MatlabMatricesWriter: Internal error in writer");
  }
  // No handling of the SCIRun errors here yet, most SCIRun functions used
  // do not use exceptions yet.
}

// Additional support functions :
// To help coordinate between the GUI in TCL and
// the functions in this module on the C++ site.
// Some of the following functions are TCL specific!

// convertdataformat
// Convert the string TCL returns into a matlabarray::mitype

matlabarray::mitype MatlabMatricesWriter::convertdataformat(std::string dataformat)
{
  matlabarray::mitype type = matlabarray::miUNKNOWN;
  if (dataformat == "same as data")  { type = matlabarray::miSAMEASDATA; }
  else if (dataformat == "double")   { type = matlabarray::miDOUBLE; }
  else if (dataformat == "single")   { type = matlabarray::miSINGLE; }
  else if (dataformat == "uint64")   { type = matlabarray::miUINT64; }
  else if (dataformat == "int64")    { type = matlabarray::miINT64; }
  else if (dataformat == "uint32")   { type = matlabarray::miUINT32; }
  else if (dataformat == "int32")    { type = matlabarray::miINT32; }
  else if (dataformat == "uint16")   { type = matlabarray::miUINT16; }
  else if (dataformat == "int16")    { type = matlabarray::miINT16; }
  else if (dataformat == "uint8")    { type = matlabarray::miUINT8; }
  else if (dataformat == "int8")     { type = matlabarray::miINT8; }
  return (type);
}

// converttcllist:
// converts a TCL formatted list into a STL array
// of strings

std::vector<std::string> MatlabMatricesWriter::converttcllist(std::string str)
{
  std::string result;
  std::vector<std::string> list(0);
  long lengthlist = 0;

  // Yeah, it is TCL dependent:
  // TCL::llength determines the length of the list
  gui->lock();
  gui->eval("llength { "+str + " }",result);	
  istringstream iss(result);
  iss >> lengthlist;
  gui->unlock();
  if (lengthlist < 0) return(list);

  list.resize(lengthlist);
  gui->lock();
  for (long p = 0;p<lengthlist;p++)
  {
    ostringstream oss;
    // TCL dependency:
    // TCL::lindex retrieves the p th element from the list
    oss << "lindex { " << str <<  " } " << p;
    gui->eval(oss.str(),result);
    list[p] = result;
  }
  gui->unlock();
  return(list);
}

// overwrite:
// Ask the user whether the file should be overwritten

bool MatlabMatricesWriter::overwrite()
{
  std::string result;
  gui->lock();
  gui->eval(id+" overwrite",result);
  gui->unlock();
  if (result == std::string("0")) 
  {
    warning("User chose to not save.");
    return(0);
  }
  return(1);
}

// displayerror:
// This function should be replaced with a more
// general function in SCIRun for displaying errors

void MatlabMatricesWriter::displayerror(std::string str)
{
  gui->lock();
  // Explicit call to TCL
  gui->execute("tk_messageBox -icon error -type ok -title {ERROR} -message {" + str + "}");
  gui->unlock();
}


} // End namespace MatlabInterface
