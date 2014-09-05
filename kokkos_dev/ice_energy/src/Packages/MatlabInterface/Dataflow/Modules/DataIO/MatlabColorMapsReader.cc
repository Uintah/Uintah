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
 * FILE: MatlabColorMapsReader.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 30 MAR 2004
 */
 
/* 
 * This module reads a matlab file and converts it to a SCIRun colormap
 *
 */

#include <sstream>
#include <string>
#include <vector>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabconverter.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/NrrdString.h>

namespace MatlabIO {

using namespace SCIRun;

class MatlabColorMapsReader : public Module 
{

public:

	// Constructor
	MatlabColorMapsReader(GuiContext*);

	// Destructor
	virtual ~MatlabColorMapsReader();

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
	//   retrieves the current filename and the colormap selected in this
	//   file and returns an object containing the full colormap
	
	matlabarray readmatlabarray(long p);

	// displayerror()
	//   Relay an error during the colormapselection process directly to
	//   the user
	
	void displayerror(std::string str);

private:

  enum { NUMPORTS = 6};
  
  // GUI variables
  GuiString				guifilename_;		// .mat filename (import from GUI)
  GuiString       guifilenameset_;
  GuiString				guicolormapinfotexts_;   	// A list of colormap-information strings of the contents of a .mat-file
  GuiString				guicolormapnames_;	// A list of colormap-names of the contents of a .mat-file 
  GuiString				guicolormapname_;		// the name of the colormap that has been selected
  GuiInt				guidisabletranspose_; // Do not convert from Fortran ordering to C++ ordering
  
  // Ports (We only use one output port)
  ColorMapOPort*			ocolormap_[NUMPORTS];
  
  // Class for translating matlab objects into SCIRun objects
  matlabconverter		translate_;
  
};

DECLARE_MAKER(MatlabColorMapsReader)

// Constructor:
// Initialise all the variables shared between TCL and SCIRun
// Only filename and colormapname are used to reconstruct the 
// settings of a previously created module
// colormapinfotexts and colormapnames serve as outputs to TCL.

MatlabColorMapsReader::MatlabColorMapsReader(GuiContext* ctx)
  : Module("MatlabColorMapsReader", ctx, Source, "DataIO", "MatlabInterface"),
    guifilename_(ctx->subVar("filename")),
    guifilenameset_(ctx->subVar("filename-set")),
    guicolormapinfotexts_(ctx->subVar("colormapinfotexts")),     
    guicolormapnames_(ctx->subVar("colormapnames")),    
	guicolormapname_(ctx->subVar("colormapname")),
	guidisabletranspose_(ctx->subVar("disable-transpose"))
{
	indexmatlabfile(false);
}

// Destructor:
// All my objects have descrutors and hence nothing needs
// explicit descruction
MatlabColorMapsReader::~MatlabColorMapsReader()
{
}

// Execute:
// Inner workings of this module
void MatlabColorMapsReader::execute()
{

      NrrdIPort *filenameport;
      if ((filenameport = static_cast<NrrdIPort *>(getIPort("filename"))))
      {
        NrrdDataHandle nrrdH;
        if (filenameport->get(nrrdH))
        {
            NrrdString fname(nrrdH);
            std::string filename = fname.getstring();
            guifilename_.set(filename);
            ctx->reset();
        }
      }



	// Get the filename from TCL.
	std::string filename = guifilename_.get();
	int disable_transpose = guidisabletranspose_.get();
	translate_.setdisabletranspose(disable_transpose);
	
	// If the filename is empty, launch an error
	if (filename == "")
	{
		error("MatlabColorMapsReader: No file name was specified");
		return;
	}

	try
	{
		for (long p=0;p<NUMPORTS;p++)
		{
	
			// Find the output port the scheduler has created 
			ocolormap_[p] = static_cast<ColorMapOPort *>(get_oport(static_cast<int>(p)));

			if(!ocolormap_[p]) 
			{
				error("MatlabColorMapsReader: Unable to initialize output port");
				return;
			}
	
			// Now read the colormap from file
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
			// creates a SCIRun colormap object
	
			SCIRun::ColorMapHandle mh;
			translate_.mlArrayTOsciColorMap(ma,mh,static_cast<SCIRun::Module *>(this));
			
			// Put the SCIRun colormap in the hands of the scheduler
			ocolormap_[p]->send(mh);
		}
	}
	
	// in case something went wrong
		
	catch (matlabfile::could_not_open_file)
	{
		error("MatlabColorMapsReader: Could not open file");
	}
	catch (matlabfile::invalid_file_format)
	{
		error("MatlabColorMapsReader: Invalid file format");
	}
	catch (matlabfile::io_error)
	{
		error("MatlabColorMapsReader: IO error");
	}
	catch (matlabfile::out_of_range)
	{
		error("MatlabColorMapsReader: Out of range");
	}
	catch (matlabfile::invalid_file_access)
	{
		error("MatlabColorMapsReader: Invalid file access");
	}
	catch (matlabfile::empty_matlabarray)
	{
		error("MatlabColorMapsReader: Empty matlab array");
	}
	catch (matlabfile::matfileerror) 
	{
		error("MatlabColorMapsReader: Internal error in reader");
	}
}


void MatlabColorMapsReader::tcl_command(GuiArgs& args, void* userdata)
{
	if(args.count() < 2){
		args.error("MatlabColorMapsReader needs a minor command");
		return;
	}

	if( args[1] == "indexmatlabfile" )
	{
		
		// It turns out that in the current design, SCIRun reads variables once
		// and then assumes they do not change and hence caches the data
		// Why it is done so is unclear to me, but in order to have interactive
		// GUIs I need to reset the context. (this synchronises the data again)
		ctx->reset();
		
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


matlabarray MatlabColorMapsReader::readmatlabarray(long p)
{
	matlabarray marray;
	std::string filename = guifilename_.get();
	std::string guicolormapname = guicolormapname_.get();
	std::string colormapname = "";
	
	// guicolormapname is a list with the name of the matrices per port
	// use the TCL command lindex to select the proper string from the list
	
	std::ostringstream oss;
	oss << "lindex {" << guicolormapname << "} " << p;
	
	gui->lock();
	gui->eval(oss.str(),colormapname);
	gui->unlock();
	
	if (colormapname == "")
	{
		// return an empty array
		return(marray);
	}
	
	if (colormapname == "<none>")
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
	marray = mfile.getmatlabarray(colormapname);
	mfile.close();

	return(marray);
}



void MatlabColorMapsReader::indexmatlabfile(bool postmsg)
{
	
	std::string filename = "";
	std::string colormapinfotexts = "";
	std::string colormapnames = "";
	std::string newcolormapname = "";
	std::string colormapname = "";
	
	guicolormapinfotexts_.set(colormapinfotexts);
	guicolormapnames_.set(colormapnames);
	
	translate_.setpostmsg(postmsg);
	
	filename = guifilename_.get();	

	if (filename == "") 
	{
		// No file has been loaded, so reset the
		// colormap name variable
		guicolormapname_.set(newcolormapname);
		return;
	}
	
	colormapname = guicolormapname_.get();
	
	std::vector<std::string> colormapnamelist(NUMPORTS);
	bool foundcolormapname[NUMPORTS];
	
	for (long p=0;p<NUMPORTS;p++)
	{
		// TCL Dependent code
		std::ostringstream oss;
		oss << "lindex { " << colormapname << " } " << p;
		gui->lock();
		gui->eval(oss.str(),colormapnamelist[p]);
		gui->unlock();
		foundcolormapname[p] = false;
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
		long cindex = 0;		// compability index, which matlab array fits the SCIRun colormap best? 
		long maxindex = 0;		// highest index found so far
			
		// Scan the file and see which matrices are compatible
		// Only those will be shown (you cannot select incompatible matrices).
			
		std::string infotext;
		
		for (long p=0;p<mfile.getnummatlabarrays();p++)
		{
			ma = mfile.getmatlabarrayinfo(p); // do not load all the data fields
			if ((cindex = translate_.sciColorMapCompatible(ma,infotext,static_cast<SCIRun::Module *>(this))))
			{
				// in case we need to propose a colormap to load, select
				// the one that is most compatible with the data
				if (cindex > maxindex) { maxindex = cindex; newcolormapname = ma.getname();}
				
				// create tcl style list to use in the array selection process
				
				colormapinfotexts += std::string("{" + infotext + "} ");
				colormapnames += std::string("{" + ma.getname() + "} "); 
				for (long q=0;q<NUMPORTS;q++)
				{
					if (ma.getname() == colormapnamelist[q]) foundcolormapname[q] = true;
				}
			}
		}

		colormapinfotexts += "{none} ";
		colormapnames += "{<none>} ";
	
		mfile.close();
	
		// automatically select a colormap if the current colormap name
		// cannot be found or if no colormapname has been specified
		
		colormapname = "";
		for (long p=0;p<NUMPORTS;p++)
		{
			if (foundcolormapname[p] == false) 
			{   
				if (p==0) 
				{
					colormapnamelist[p] = newcolormapname;
				}
				else
				{
					colormapnamelist[p] = "<none>";
				}
			}
			colormapname += "{" + colormapnamelist[p] + "} ";
		}
		
		// Update TCL on the contents of this colormap
		guicolormapname_.set(colormapname);
		guicolormapinfotexts_.set(colormapinfotexts);
		guicolormapnames_.set(colormapnames);
	}
	
	// in case something went wrong
	// close the file and then dermine the problem

	catch (matlabfile::could_not_open_file)
	{
		displayerror("MatlabColorMapsReader: Could not open file");
	}
	catch (matlabfile::invalid_file_format)
	{
		displayerror("MatlabColorMapsReader: Invalid file format");
	}
	catch (matlabfile::io_error)
	{
		displayerror("MatlabColorMapsReader: IO error");
	}
	catch (matlabfile::out_of_range)
	{
		displayerror("MatlabColorMapsReader: Out of range");
	}
	catch (matlabfile::invalid_file_access)
	{
		displayerror("MatlabColorMapsReader: Invalid file access");
	}
	catch (matlabfile::empty_matlabarray)
	{
		displayerror("MatlabColorMapsReader: Empty matlab array");
	}
	catch (matlabfile::matfileerror) 
	{
		displayerror("MatlabColorMapsReader: Internal error in reader");
	}
	return;
}


void MatlabColorMapsReader::displayerror(std::string str)
{
  gui->lock();
  // Explicit call to TCL
  gui->execute("tk_messageBox -icon error -type ok -title {ERROR} -message {" + str + "}");
  gui->unlock();
}


} // End namespace MatlabIO
