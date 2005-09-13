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
 *  UnuMake.cc:  Create a nrrd (or nrrd header) from scratch.
 *
 *  Written by:
 *   Darby Van Uitert
 *   February 2004
 *
 */  

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <sys/stat.h>

#include <Dataflow/Ports/NrrdPort.h>

/* bad Gordon */
extern "C" {
  int _nrrdReadNrrdParse_keyvalue(Nrrd *nrrd, NrrdIoState *io, int useBiff);
}

namespace SCITeem {

using namespace SCIRun;

class UnuMake : public Module {
public:
  //! Constructor
  UnuMake(GuiContext*);

  //! Destructor
  virtual ~UnuMake();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  void read_file_and_create_nrrd();
  int  parse_gui_vars();

  //! Gui variables
  GuiString       filename_;
  GuiString       header_filename_;
  GuiInt          write_header_;
  GuiString       data_type_;
  GuiString       samples_;
  GuiString       spacing_;
  GuiString       labels_;
  GuiString       kind_;
  GuiString       content_;
  GuiInt          line_skip_;
  GuiInt          byte_skip_;
  GuiString       endian_;
  GuiString       encoding_;
  GuiString       key1_;
  GuiString       key2_;
  GuiString       key3_;
  GuiString       val1_;
  GuiString       val2_;
  GuiString       val3_;

  NrrdData*       nout_;
  NrrdIoState*    nio_;
  NrrdDataHandle  read_handle_;

  string          old_filename_;
  time_t          old_filemodification_;
  vector<int>     sz_;
  vector<double>  sp_;
  vector<string>  lb_;
  int             dimension_;
};

} // end namespace SCITeem



using namespace SCITeem;

DECLARE_MAKER(UnuMake)

UnuMake::UnuMake(GuiContext* ctx)
  : Module("UnuMake", ctx, Source, "UnuAtoM", "Teem"),
    filename_(ctx->subVar("filename")),
    header_filename_(ctx->subVar("header_filename")),
    write_header_(ctx->subVar("write_header")),
    data_type_(ctx->subVar("data_type")),
    samples_(ctx->subVar("samples")),
    spacing_(ctx->subVar("spacing")),
    labels_(ctx->subVar("labels")),
    kind_(ctx->subVar("kind")),
    content_(ctx->subVar("content")),
    line_skip_(ctx->subVar("line_skip")),
    byte_skip_(ctx->subVar("byte_skip")),
    endian_(ctx->subVar("endian")),
    encoding_(ctx->subVar("encoding")),
    key1_(ctx->subVar("key1")),
    key2_(ctx->subVar("key2")),
    key3_(ctx->subVar("key3")),
    val1_(ctx->subVar("val1")),
    val2_(ctx->subVar("val2")),
    val3_(ctx->subVar("val3")),
    read_handle_(0),
    old_filemodification_(0),
    dimension_(0)
{
  // set endianness to be that of the current machine for default
  if (airMyEndian == airEndianLittle) {
    endian_.set("little");
  } else {
    endian_.set("big");
  }
}

UnuMake::~UnuMake(){
}


void UnuMake::read_file_and_create_nrrd() {
  // Reset errors in case there were any from
  // a previous attempt.
  update_msg_state(Reset);

  filename_.reset();
  string fn(filename_.get());
  if(fn == "") {
    error("No data filename specified.");
    return;
  }

  // Read the status of this file so we can compare modification timestamps.
  struct stat buf;
  if (stat(fn.c_str(), &buf) == -1) {
    error(string("UnuMake error - file not found: '")+fn+"'");
    return;
  }

  // If we haven't read yet, or if it's a new filename, 
  //  or if the datestamp has changed -- then read...
#ifdef __sgi
  time_t new_filemodification = buf.st_mtim.tv_sec;
#else
  time_t new_filemodification = buf.st_mtime;
#endif
//   if(!read_handle_.get_rep() || 
//      fn != old_filename_ || 
//      new_filemodification != old_filemodification_)
//   {
  if (1) { // read automatically because one of the many ui values could have changed
    old_filemodification_ = new_filemodification;
    old_filename_=fn;
    read_handle_ = 0;

    nout_ = scinew NrrdData();
    nio_ = nrrdIoStateNew();
    
    
    // Reset guivars
    write_header_.reset();
    data_type_.reset();
    kind_.reset();
    line_skip_.reset();
    byte_skip_.reset();

    // set first axes kind to be what user specified
    string kind = kind_.get();
    if (kind == "nrrdKindDomain") 
      nout_->nrrd->axis[0].kind = nrrdKindDomain;
    else if (kind == "nrrdKindScalar")
      nout_->nrrd->axis[0].kind = nrrdKindScalar;
    else if (kind == "nrrdKind3Color")
      nout_->nrrd->axis[0].kind = nrrdKind3Color;
    else if (kind == "nrrdKind3Vector")
      nout_->nrrd->axis[0].kind = nrrdKind3Vector;
    else if (kind == "nrrdKind3Normal")
      nout_->nrrd->axis[0].kind = nrrdKind3Normal;
    else if (kind == "nrrdKind3DSymMatrix")
      nout_->nrrd->axis[0].kind = nrrdKind3DSymMatrix;
    else if (kind == "nrrdKind3DMaskedSymMatrix")
      nout_->nrrd->axis[0].kind = nrrdKind3DMaskedSymMatrix;
    else if (kind == "nrrdKind3DMatrix")
      nout_->nrrd->axis[0].kind = nrrdKind3DMatrix;
    else if (kind == "nrrdKindList")
      nout_->nrrd->axis[1].kind = nrrdKindList;
    else if (kind == "nrrdKindStub")
      nout_->nrrd->axis[1].kind = nrrdKindStub;
    else
      nout_->nrrd->axis[0].kind = nrrdKindUnknown;

    
    string data_type = data_type_.get();
    if (data_type == "char") {
      nout_->nrrd->type = nrrdTypeChar;
    } else if (data_type == "unsigned char") {
      nout_->nrrd->type = nrrdTypeUChar;
    } else if (data_type == "short") {
      nout_->nrrd->type = nrrdTypeShort;
    } else if (data_type == "unsigned short") {
      nout_->nrrd->type = nrrdTypeUShort;
    } else if (data_type == "int") {
      nout_->nrrd->type = nrrdTypeInt;
    } else if (data_type == "unsigned int") {
      nout_->nrrd->type = nrrdTypeUInt;
    } else if (data_type == "long long") {
      nout_->nrrd->type = nrrdTypeLLong;
    } else if (data_type == "unsigned long long") {
      nout_->nrrd->type = nrrdTypeULLong;
    } else if (data_type == "float") {
      nout_->nrrd->type = nrrdTypeFloat;
    } else if (data_type == "double") {
      nout_->nrrd->type = nrrdTypeDouble;
    } else {
      error("Unkown data type");
      return;
    }
    
    // Parse samples input for values and also get dimension.
    // Any errors will be output in the parse_gui_vars function
    if(parse_gui_vars() == 0) {
      return;
    }
    
    nout_->nrrd->dim = dimension_;

    // set the rest of the axes to have a kind of nrrdKindDomain
    for (int i=1; i<dimension_; i++) {
      nout_->nrrd->axis[i].kind = nrrdKindDomain;
    }
    
    // Set the nrrd's labels, sizes, and spacings that were
    // parsed in the parse_gui_vars function
    for(int i=0; i<dimension_; i++) {
      nout_->nrrd->axis[i].label = airStrdup(lb_[i].c_str());
      nout_->nrrd->axis[i].size = sz_[i];
      nout_->nrrd->axis[i].spacing = sp_[i];
      nrrdAxisInfoMinMaxSet(nout_->nrrd, i, nrrdCenterUnknown);
    }
    
    nout_->nrrd->content = airStrdup(content_.get().c_str());
    
    // Key/Value pairs
    string t_key = airOneLinify((char*)key1_.get().c_str());
    string t_val = airOneLinify((char*)val1_.get().c_str());
    if (t_key != "" && t_val != "") {
      string temp = "\"" + t_key + ":=" + t_val + "\"";
      nio_->line = (char*)temp.c_str();
      if(_nrrdReadNrrdParse_keyvalue(nout_->nrrd, nio_, AIR_TRUE)) {
	string err = biffGetDone(NRRD);
	error(err);
	nio_->line = NULL;
      }
      nio_->line = NULL;
    }
    
    t_key = airOneLinify((char*)key2_.get().c_str());
    t_val = airOneLinify((char*)val2_.get().c_str());
    if (t_key != "" && t_val != "") {
      string temp = "\"" + t_key + ":=" + t_val + "\"";
      nio_->line = (char*)temp.c_str();
      if(_nrrdReadNrrdParse_keyvalue(nout_->nrrd, nio_, AIR_TRUE)) {
	string err = biffGetDone(NRRD);
	error(err);
	nio_->line = NULL;
      }
      nio_->line = NULL;
    }
    
    t_key = airOneLinify((char*)key3_.get().c_str());
    t_val = airOneLinify((char*)val3_.get().c_str());
    if (t_key != "" && t_val != "") {
      string temp = "\"" + t_key + ":=" + t_val + "\"";
      nio_->line = (char*)temp.c_str();
      if(_nrrdReadNrrdParse_keyvalue(nout_->nrrd, nio_, AIR_TRUE)) {
	string err = biffGetDone(NRRD);
	error(err);
	nio_->line = NULL;
      }
      nio_->line = NULL;
    }
    
    // Case for generating a header
    if(write_header_.get()) {
      nio_->lineSkip = line_skip_.get();
      nio_->byteSkip = byte_skip_.get();
      
      string encoding = encoding_.get();
      if (encoding == "Raw") {
	nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeRaw];
      } else if (encoding == "ASCII") {
	nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeAscii];
      } else if (encoding == "Hex") {
	nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeHex];
      } else if (encoding == "Gzip") {
	nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeGzip];
      } else if (encoding == "Bzip2") {
	nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeBzip2];
      }  else {
	error("Unknown encoding type");
	return;
      }

      /* start new */      
      nio_->dataFileDim = nout_->nrrd->dim;
      airArraySetLen(nio_->dataFNArr, 1);
      nio_->dataFN[0] = airStrdup(fn.c_str());
      /* end new */

      nio_->detachedHeader = AIR_TRUE;
      nio_->skipData = AIR_TRUE;
      
      if (endian_.get() == "little") {
	nio_->endian = airEndianLittle;
      } else {
	nio_->endian = airEndianBig;
      }
      
      FILE* fileOut;
      if(header_filename_.get() == "") {
	remark("No header filename spcified.  Attempting to write header in directory of original data file.");
	string out = fn + ".nhdr";
	if (!(fileOut = airFopen(out.c_str(),stdout, "wb"))) {
	  error("Error opening header file for writing.");
	  return;
	} else {
	  nrrdFormatNRRD->write(fileOut, nout_->nrrd, nio_);
	  airFclose(fileOut);
	}
      } else {
	if (!(fileOut = airFopen(header_filename_.get().c_str(),stdout, "wb"))) {
	  error("Error opening header file for writing.");
	  return;
	} else {
	  nrrdFormatNRRD->write(fileOut, nout_->nrrd, nio_);
	  airFclose(fileOut);
	}
      }
    } 
    
    // Already written header so reset as if we
    // haven't written it out and still need to
    // read the data
    nrrdIoStateInit(nio_);
    nio_->detachedHeader = AIR_FALSE;
    nio_->skipData = AIR_FALSE;
    
    nio_->lineSkip = line_skip_.get();
    nio_->byteSkip = byte_skip_.get();
    
    string encoding = encoding_.get();
    if (encoding == "Raw") {
      nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeRaw];
    } else if (encoding == "ASCII") {
      nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeAscii];
    } else if (encoding == "Hex") {
      nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeHex];
    } else if (encoding == "Gzip") {
      nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeGzip];
    } else if (encoding == "Bzip2") {
      nio_->encoding = nrrdEncodingArray[nrrdEncodingTypeBzip2];
    }  else {
      error("Unknown encoding type");
      return;
    }
    
    FILE *dataFile;
    // Assume only reading in a single file
    if(!(dataFile = airFopen(fn.c_str(), stdin, "rb") )) {
      error("Couldn't open file " + fn + " for reading.");
      return;
    }
    if(nrrdLineSkip(dataFile, nio_)) {
      error("Couldn't skip lines.");
      airFclose(dataFile);
      return;
    } 
    if(!nio_->encoding->isCompression) {
      if(nrrdByteSkip(dataFile, nout_->nrrd, nio_)) {
	error("Couldn't skip bytes.");
	airFclose(dataFile);
	return;
      }
    }
    nout_->nrrd->data = calloc(nrrdElementNumber(nout_->nrrd), 
			       nrrdElementSize(nout_->nrrd));
    if (NULL == nout_->nrrd->data) {
      error("Allocation failure");
      return;
    }
    if (nio_->encoding->read(dataFile, nout_->nrrd->data, 
			     nrrdElementNumber(nout_->nrrd), 
			     nout_->nrrd, nio_)) {
      error("Error reading data.");
      string err = biffGetDone(NRRD);
      error(err);
      airFclose(dataFile);
      return;
    } 
    
    airFclose(dataFile);
    
    if(1 < nrrdElementSize(nout_->nrrd)
       && nio_->encoding->endianMatters
       && nio_->endian != AIR_ENDIAN) {
      nrrdSwapEndian(nout_->nrrd);
    }
    
    read_handle_ = nout_;
  } else {
    dimension_ = 0;
  }
}

int UnuMake::parse_gui_vars() {
  lb_.clear();
  sz_.clear();
  sp_.clear();
  dimension_ = 0;

  samples_.reset();
  spacing_.reset();
  labels_.reset();

  char ch;
  int i=0, start=0, end=0, which=0,counter=0;;
  bool inword = false;

  // Determine the dimension based on the samples string
  while (i < (int)samples_.get().length()) {
    ch = samples_.get()[i];
    if(isspace(ch)) {
      if (inword) {
	dimension_++;
	inword = false;
      }
    } else if (i == (int)samples_.get().length()-1) {
      dimension_++;
      inword = false;
    } else {
      if(!inword) 
	inword = true;
    }
    i++;
  }

  if (dimension_ == 0) {
    error("Dimensions appears to be 0 based on the sample values.");
    return 0;
  }

  // The samples, spacing and labels should all contain the same
  // number of "entries"
  
  // Size/samples
  i=0, which = 0, start=0, end=0, counter=0;
  inword = false;
  while (i < (int)samples_.get().length()) {
    ch = samples_.get()[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	sz_.push_back(atoi(samples_.get().substr(start, end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)samples_.get().length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      sz_.push_back(atoi(samples_.get().substr(start, end-start).c_str()));
      which++;
      counter++;
      inword = false;
    } else {
      if(!inword) {
	start = i;
	inword = true;
      }
    }
    i++;
  }

  if(counter != dimension_) {
    error("Number of samples specified incorrectly.");
    dimension_ = 0;
    return 0;
  }

  // Spacing
  i=0, which = 0, start=0, end=0, counter=0;
  inword = false;
  while (i < (int)spacing_.get().length()) {
    ch = spacing_.get()[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	sp_.push_back(atof(spacing_.get().substr(start, end-start).c_str()));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)spacing_.get().length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      sp_.push_back(atof(spacing_.get().substr(start, end-start).c_str()));
      which++;
      counter++;
      inword = false;
    } else {
      if(!inword) {
	start = i;
	inword = true;
      }
    }
    i++;
  } 

  if(counter != dimension_) {
    error("Number of spacing values given does not match the number of sample values given.");
    dimension_ = 0;
    return 0;
  }


  // Labels
  i=0, which = 0, start=0, end=0, counter=0;
  inword = false;
  while (i < (int)labels_.get().length()) {
    ch = labels_.get()[i];
    if(isspace(ch)) {
      if (inword) {
	end = i;
	lb_.push_back(labels_.get().substr(start, end-start));
	which++;
	counter++;
	inword = false;
      }
    } else if (i == (int)labels_.get().length()-1) {
      if (!inword) {
	start = i;
      }
      end = i+1;
      lb_.push_back(labels_.get().substr(start, end-start));
      which++;
      counter++;
      inword = false;
    } else {
      if(!inword) {
	start = i;
	inword = true;
      }
    }
    i++;
 } 

  if(counter != dimension_) {
    remark("Labels specified incorrectly. Setting label as empty string for unspecified labels.");
  }

   for(int i=counter; i<dimension_; i++) {
     lb_.push_back("");
   }

  // return the dimension
  return dimension_;
}

void UnuMake::execute()
{
  update_state(NeedData);
  
  //get_nrrd_info(read_handle_);
  read_file_and_create_nrrd(); 
  
  // Do nothing if dimension_ is 0. This happens if the
  // gui entries are incorrect. Errors will be printed
  // from the read_file_and_create_nrrd function.
  if (dimension_ == 0) 
    return;
  
  if (!read_handle_.get_rep()) { 
    error("Please generate a Nrrd.");
    return; 
  }

  // Send the data downstream.
  NrrdOPort *outport = (NrrdOPort *)get_oport("OutputNrrd");
  outport->send(read_handle_);

  update_state(Completed);
}


void UnuMake::tcl_command(GuiArgs& args, void* userdata)
{

    Module::tcl_command(args, userdata);


} // End namespace Teem


