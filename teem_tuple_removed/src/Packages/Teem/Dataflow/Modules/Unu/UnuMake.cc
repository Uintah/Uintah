/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <Dataflow/share/share.h>
#include <Core/GuiInterface/GuiVar.h>
#include <sys/stat.h>

#include <Teem/Dataflow/Ports/NrrdPort.h>

/* bad Gordon */
extern "C" {
  int _nrrdReadNrrdParse_keyvalue(Nrrd *nrrd, NrrdIO *io, int useBiff);
}

namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE UnuMake : public Module {
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
  NrrdIO*         nio_;
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
  : Module("UnuMake", ctx, Source, "Unu", "Teem"),
    filename_(ctx->subVar("filename")),
    header_filename_(ctx->subVar("header_filename")),
    write_header_(ctx->subVar("write_header")),
    data_type_(ctx->subVar("data_type")),
    samples_(ctx->subVar("samples")),
    spacing_(ctx->subVar("spacing")),
    labels_(ctx->subVar("labels")),
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
  if (stat(fn.c_str(), &buf)) {
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
  if(!read_handle_.get_rep() || 
     fn != old_filename_ || 
     new_filemodification != old_filemodification_)
  {
    old_filemodification_ = new_filemodification;
    old_filename_=fn;
    read_handle_ = 0;

    nout_ = scinew NrrdData();
    nio_ = nrrdIONew();
    
    
    // Reset guivars
    write_header_.reset();
    data_type_.reset();
    line_skip_.reset();
    byte_skip_.reset();
    
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
    
    // Set the nrrd's labels, sizes, and spacings that were
    // parsed in the parse_gui_vars function
    for(int i=0; i<dimension_; i++) {
      nout_->nrrd->axis[i].label = strdup(lb_[i].c_str());
      nout_->nrrd->axis[i].size = sz_[i];
      nout_->nrrd->axis[i].spacing = sp_[i];
      nrrdAxisMinMaxSet(nout_->nrrd, i, nrrdCenterUnknown);
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
	nio_->encoding = nrrdEncodingArray[1];
      } else if (encoding == "ASCII") {
	nio_->encoding = nrrdEncodingArray[2];
      } else if (encoding == "Hex") {
	nio_->encoding = nrrdEncodingArray[3];
      } else if (encoding == "Gzip") {
	nio_->encoding = nrrdEncodingArray[4];
      } else if (encoding == "Bzip2") {
	nio_->encoding = nrrdEncodingArray[5];
      }  else {
	error("Non-existant encoding type");
	return;
      }
      
      nio_->dataFN = airStrdup(fn.c_str());
      
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
	  AIR_FCLOSE(fileOut);
	}
      } else {
	if (!(fileOut = airFopen(header_filename_.get().c_str(),stdout, "wb"))) {
	  error("Error opening header file for writing.");
	  return;
	} else {
	  nrrdFormatNRRD->write(fileOut, nout_->nrrd, nio_);
	  AIR_FCLOSE(fileOut);
	}
      }
    } 
    
    // All ready written header so reset as if we
    // haven't written it out and still need to
    // read the data
    nio_->detachedHeader = AIR_FALSE;
    nio_->skipData = AIR_FALSE;
    
    nrrdIOInit(nio_);
    nio_->lineSkip = line_skip_.get();
    nio_->byteSkip = byte_skip_.get();
    
    string encoding = encoding_.get();
    if (encoding == "Raw") {
      nio_->encoding = nrrdEncodingArray[1];
    } else if (encoding == "ASCII") {
      nio_->encoding = nrrdEncodingArray[2];
    } else if (encoding == "Hex") {
      nio_->encoding = nrrdEncodingArray[3];
    } else if (encoding == "Gzip") {
      nio_->encoding = nrrdEncodingArray[4];
    } else if (encoding == "Bzip2") {
      nio_->encoding = nrrdEncodingArray[5];
    }  else {
      error("Non-existant encoding type");
      return;
    }
    
    
    // Assume only reading in a single file
    if(!(nio_->dataFile = airFopen(fn.c_str(), stdin, "rb") )) {
      error("Couldn't open file " + fn + " for reading.");
      return;
    }
    if(nrrdLineSkip(nio_)) {
      error("Couldn't skip lines.");
      AIR_FCLOSE(nio_->dataFile);
      return;
    } 
    if(!nio_->encoding->isCompression) {
      if(nrrdByteSkip(nout_->nrrd,nio_)) {
	error("Couldn't skip bytes.");
	AIR_FCLOSE(nio_->dataFile);
	return;
      }
    }
    if (nio_->encoding->read(nout_->nrrd, nio_)) {
      error("Error reading data.");
      string err = biffGetDone(NRRD);
      error(err);
      AIR_FCLOSE(nio_->dataFile);
      return;
    } 
    
    AIR_FCLOSE(nio_->dataFile);
    
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
  if (!outport) {
    error("Unable to initialize oport 'OutportNrrd'.");
    return;
  }
  outport->send(read_handle_);

  update_state(Completed);
}


void UnuMake::tcl_command(GuiArgs& args, void* userdata)
{

    Module::tcl_command(args, userdata);


} // End namespace Teem


