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


//    File   : HDF5DataReader.h
//    Author : Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : May 2003

#if !defined(HDF5DataReader_h)
#define HDF5DataReader_h

#include <Dataflow/Network/Module.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/StringPort.h>

#include <Core/GuiInterface/GuiVar.h>

#include <Packages/DataIO/share/share.h>

namespace DataIO {

using namespace SCIRun;

#define MAX_PORTS 8
#define MAX_DIMS 6

class DataIOSHARE HDF5DataReader : public Module {
protected:
  enum { MERGE_NONE=0,   MERGE_LIKE=1,   MERGE_TIME=2 };

public:
  HDF5DataReader(GuiContext *context);

  virtual ~HDF5DataReader();

  virtual void execute();

  void ReadandSendData( string& filename,
			vector< string >& pathList,
			vector< string >& datasethList,
			bool cache,
			int which );

  void parseDatasets( string datasets,
		      vector<string>& pathhList,
		      vector<string>& datasethList );

  unsigned int parseAnimateDatasets( vector<string>& pathhList,
				     vector<string>& datasethList,
				     vector< vector<string> >& frame_paths,
				     vector< vector<string> >& frame_datasets );

  vector<int> getDatasetDims( string filename, string group, string dataset );

  //  float* readGrid( string filename );
  //  float* readData( string filename );
  NrrdDataHandle readDataset( string filename, string path, string dataset );

  string getDumpFileName( string filename );
  bool checkDumpFile( string filename, string dumpname );
  int createDumpFile( string filename, string dumpname );

  virtual void tcl_command(GuiArgs&, void*);

protected:
  bool animate_execute( string new_filename,
			vector< vector<string> >& frame_paths,
			vector< vector<string> >& frame_datasets );
			

  int increment(int which, int lower, int upper);

  bool is_mergeable(NrrdDataHandle h1, NrrdDataHandle h2);

protected:
  GuiInt         gui_power_app_;
  GuiString      gui_power_app_cmd_;

  GuiFilename    gui_filename_;
  GuiString      gui_datasets_;
  GuiString      gui_dumpname_;
  GuiString      gui_ports_;

  GuiInt         gui_ndims_;

  GuiInt         gui_merge_data_;
  GuiInt         gui_assume_svt_;
  GuiInt         gui_animate_;

  GuiString      gui_animate_frame_;
  GuiString	 gui_animate_tab_;
  GuiString	 gui_basic_tab_;
  GuiString	 gui_extended_tab_;
  GuiString	 gui_playmode_tab_;

  GuiDouble      gui_selectable_min_;
  GuiDouble      gui_selectable_max_;
  GuiInt         gui_selectable_inc_;
  GuiInt         gui_range_min_;
  GuiInt         gui_range_max_;
  GuiString      gui_playmode_;
  GuiInt         gui_current_;
  GuiString      gui_execmode_;
  GuiInt         gui_delay_;
  GuiInt         gui_inc_amount_;
  //update_type_ must be declared after selectable_max_ which is
  //traced in the tcl code. If update_type_ is set to Auto having it
  //last will prevent the net from executing when it is instantiated.
  GuiString      gui_update_type_;
  int            inc_;

  vector< GuiInt* > gui_dims_;
  vector< GuiInt* > gui_starts_;
  vector< GuiInt* > gui_counts_;
  vector< GuiInt* > gui_strides_;

  string old_filename_;
  string old_datasets_;
  time_t old_filemodification_;

  string sel_filename_;
  string sel_datasets_;
  time_t sel_filemodification_;

  int dims_[MAX_DIMS];
  int starts_[MAX_DIMS];
  int counts_[MAX_DIMS];
  int strides_[MAX_DIMS];

  NrrdDataHandle nrrd_output_handles_[MAX_PORTS];
  MatrixHandle matrix_output_handle_;

  bool loop_;
  bool execute_error_;
};


} // end namespace SCIRun

#endif // HDF5DataReader_h
