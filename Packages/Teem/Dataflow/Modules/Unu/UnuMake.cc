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
  void get_nrrd_info(NrrdDataHandle handle);
  void read_file_and_create_nrrd();
  int  parse_gui_vars();

  //! Gui variables
  GuiString       label_;
  GuiString       type_;
  GuiString       axis_;
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
  NrrdDataHandle  send_handle_;

  string          old_filename_;
  time_t          old_filemodification_;
  int             cached_label_generation_;
  char *          cached_label_;
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
    label_(ctx->subVar("label")),
    type_(ctx->subVar("type")),
    axis_(ctx->subVar("axis")),
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
    send_handle_(0),
    old_filemodification_(0),
    cached_label_generation_(0),
    cached_label_(0),
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
  if (cached_label_) { delete [] cached_label_; cached_label_ = 0; }
}

void UnuMake::get_nrrd_info(NrrdDataHandle handle)
{
  // If nrrd information hasn't been generated from gui
  // entries, generate it
  if (!handle.get_rep()) { 
    read_file_and_create_nrrd(); 
    // Do nothing if dimension_ is 0. This happens if the
    // gui entries are incorrect.
    if (dimension_ == 0)
      return;
    handle = read_handle_;
    
  }

  // Clear any old info.
  ostringstream clear; 
  clear << id.c_str() << " clear_axis_info";
  gui->execute(clear.str());
  
  if (!handle.get_rep()) { return; }
  // Send the axis info to the gui.

  const string addnew =
    id + " add_axis_info CreateNewTuple FromBelow Unknown --- --- --- ---";
  gui->execute(addnew);

  // Call the following tcl method:
  // add_axis_info {id label center size spacing min max}
  for (int i = 0; i < handle->nrrd->dim; i++) {
    ostringstream add; 
    add << id.c_str() << " add_axis_info ";
    add << i << " ";
    if (!handle->nrrd->axis[i].label) {
      handle->nrrd->axis[i].label = strdup("---");
    }
    add << handle->nrrd->axis[i].label << " ";
    switch (handle->nrrd->axis[i].center) {
    case nrrdCenterUnknown :
      add << "Unknown ";
      break;
    case nrrdCenterNode :
      add << "Node ";
      break;
    case nrrdCenterCell :
      add << "Cell ";
      break;
    }
    add << handle->nrrd->axis[i].size << " ";
    add << handle->nrrd->axis[i].spacing << " ";

    if (!(AIR_EXISTS(handle->nrrd->axis[i].min) && 
	  AIR_EXISTS(handle->nrrd->axis[i].max)))
      nrrdAxisMinMaxSet(handle->nrrd, i, nrrdCenterNode);
    
    add << handle->nrrd->axis[i].min << " ";
    add << handle->nrrd->axis[i].max << endl;  
    gui->execute(add.str());
  }
}


void UnuMake::read_file_and_create_nrrd() {
  // Reset errors in case there were any from
  // a previous attempt.
  update_msg_state(Reset);

  filename_.reset();
  if(filename_.get() == "") {
    error("No data filename specified.");
    return;
  }
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

    nio_->dataFN = airStrdup(filename_.get().c_str());

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
      string out = filename_.get() + ".nhdr";
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
  if(!(nio_->dataFile = airFopen(filename_.get().c_str(), stdin, "rb") )) {
    error("Couldn't open file " + filename_.get() + " for reading.");
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
  } else {
    cerr << "Read data ok.\n";
  }

  AIR_FCLOSE(nio_->dataFile);
  
  if(1 < nrrdElementSize(nout_->nrrd)
     && nio_->encoding->endianMatters
     && nio_->endian != AIR_ENDIAN) {
    nrrdSwapEndian(nout_->nrrd);
  }

  read_handle_ = nout_;

  // Display the nrrd info for the
  // user to select a tuple axis
  get_nrrd_info(read_handle_);
  
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
    remark("Labels specified incorrectly. Setting label as 'unknown' for unspecified labels.");
  }

   for(int i=counter; i<dimension_; i++) {
     lb_.push_back("unknown");
   }

  // return the dimension
  return dimension_;
}

void UnuMake::execute()
{
  update_state(NeedData);

  get_nrrd_info(read_handle_);

  if (!read_handle_.get_rep()) { 
    error("Please generate a Nrrd, and set up the axes for it.");
    return; 
  }

  axis_.reset();
  if (axis_.get() == "") {
    error("Please select the axis which is tuple from the UI");
    return;
  }

  // Compute which axis was picked.
  string ax(axis_.get());
  int axis = 0;
  if (ax.size()) {
    axis = atoi(ax.substr(4).c_str()); // Trim 'axis' from the string.
  }

  if (cached_label_generation_ == read_handle_->generation &&
      (cached_label_ == 0 ||
       strcmp(read_handle_->nrrd->axis[0].label, cached_label_) != 0))
  {
    if (read_handle_->nrrd->axis[0].label)
    {
      delete [] read_handle_->nrrd->axis[0].label;
      read_handle_->nrrd->axis[0].label = 0;
    }
    if (cached_label_)
    {
      read_handle_->nrrd->axis[0].label = strdup(cached_label_);
    }
  }
  
  bool added_tuple_axis = false;
  if (ax == "axisCreateNewTuple" && !added_tuple_axis)
  {
    // do add permute work here.
    Nrrd *pn = nrrdNew();
    if (nrrdAxesInsert(pn, read_handle_->nrrd, 0))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error adding a tuple axis: ") + err);
      free(err);
      return;
    }
    NrrdData *newnrrd = new NrrdData();
    newnrrd->nrrd = pn;
    newnrrd->copy_sci_data(*read_handle_.get_rep());
    send_handle_ = newnrrd;
    added_tuple_axis = true;
  }
  else if (axis != 0)
  {
    // Permute so that 0 is the tuple axis.
    const int sz = read_handle_->nrrd->dim;
    int perm[NRRD_DIM_MAX];
    Nrrd *pn = nrrdNew();
    // Init the perm array.
    for(int i = 0; i < sz; i++)
    {
      perm[i] = i;
    }

    // Swap the selected axis with 0.
    perm[0] = axis;
    perm[axis] = 0;

    if (nrrdAxesPermute(pn, read_handle_->nrrd, perm))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error adding a tuple axis: ") + err);
      free(err);
      return;
    }
    NrrdData *newnrrd = new NrrdData();
    newnrrd->nrrd = pn;
    newnrrd->copy_sci_data(*read_handle_.get_rep());
    send_handle_ = newnrrd;
  }
  else
  {
    send_handle_ = read_handle_;
  }

  // If the tuple label is valid use it. If not use the string provided
  // in the gui.
  vector<string> elems;
  if (added_tuple_axis || (! send_handle_->get_tuple_indecies(elems)))
  {
    int axis_size = send_handle_->nrrd->axis[0].size;

    // Set tuple axis name.
    label_.reset();
    type_.reset();
    string label(label_.get() + ":" + type_.get());

    string full_label = label;
    int count;
    if (type_.get() == "Scalar") count=axis_size-1;
    else if (type_.get() == "Vector") count=axis_size/3-1;
    else /* if (type_.get() == "Tensor") */ count=axis_size/7-1;
    while (count > 0) {
      full_label += string("," + label);
      count--;
    }
    // Cache off a copy of the prior label in case of axis change
    // later.
    if (send_handle_.get_rep() == read_handle_.get_rep())
    {
      if (cached_label_) { delete [] cached_label_; cached_label_ = 0; }
      if (read_handle_->nrrd->axis[0].label)
      {
	cached_label_ = strdup(read_handle_->nrrd->axis[0].label);
      }
      cached_label_generation_ = read_handle_->generation;
    }

    // TODO:  This appears to memory leak the existing label string.
    send_handle_->nrrd->axis[0].label = strdup(full_label.c_str());
  }

  // Send the data downstream.
  NrrdOPort *outport = (NrrdOPort *)get_oport("OutputNrrd");
  if (!outport) {
    error("Unable to initialize oport 'OutportNrrd'.");
    return;
  }
  outport->send(send_handle_);

  update_state(Completed);
}


void UnuMake::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("UnuMake needs a minor command");
    return;
  }
  else if (args[1] == "generate_nrrd") {
    read_file_and_create_nrrd();
  }
  else
  {
    Module::tcl_command(args, userdata);
  }


} // End namespace Teem


