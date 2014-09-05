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

#include <Packages/Teem/Core/Datatypes/NrrdData.h>
#include <Packages/Teem/Dataflow/Ports/NrrdPort.h>

#include <Dataflow/Ports/MatrixPort.h>

#include <Core/GuiInterface/GuiVar.h>

#include <Packages/DataIO/share/share.h>

namespace DataIO {

using namespace SCIRun;
using namespace SCITeem;

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
			vector< string >& paths,
			vector< string >& datasets,
			bool cache,
			int which );

  void parseDatasets( string new_datasets,
		      vector<string>& paths,
		      vector<string>& datasets );

  unsigned int parseAnimateDatasets( vector<string>& paths,
				     vector<string>& datasets,
				     vector< vector<string> >& frame_paths,
				     vector< vector<string> >& frame_datasets );

  vector<int> getDatasetDims( string filename, string group, string dataset );

  //  float* readGrid( string filename );
  //  float* readData( string filename );
  NrrdDataHandle readDataset( string filename, string path, string dataset );

  virtual void tcl_command(GuiArgs&, void*);

protected:
  void animate_execute( string new_filename,
			vector< vector<string> >& frame_paths,
			vector< vector<string> >& frame_datasets );
			

  int increment(int which, int lower, int upper);

protected:
  GuiDouble      selectable_min_;
  GuiDouble      selectable_max_;
  GuiInt         selectable_inc_;
  GuiInt         range_min_;
  GuiInt         range_max_;
  GuiString      playmode_;
  GuiInt         current_;
  GuiString      execmode_;
  GuiInt         delay_;
  GuiInt         inc_amount_;
  int            inc_;
  int            last_input_;
  NrrdDataHandle last_output_;

  bool is_mergeable(NrrdDataHandle h1, NrrdDataHandle h2) const;

  GuiFilename filename_;
  GuiString datasets_;
  GuiString dumpname_;
  GuiString ports_;

  GuiInt nDims_;

  GuiInt mergeData_;
  GuiInt assumeSVT_;

  GuiInt animate_;

  vector< GuiInt* > gDims_;
  vector< GuiInt* > gStarts_;
  vector< GuiInt* > gCounts_;
  vector< GuiInt* > gStrides_;

  string old_filename_;
  string old_datasets_;
  time_t old_filemodification_;

  string tmp_filename_;
  string tmp_datasets_;
  time_t tmp_filemodification_;

  int mergedata_;
  int assumesvt_;

  int dims_[MAX_DIMS];
  int starts_[MAX_DIMS];
  int counts_[MAX_DIMS];
  int strides_[MAX_DIMS];

  NrrdDataHandle nHandles_[MAX_PORTS];
  MatrixHandle mHandle_;

  bool loop_;
  bool error_;
};


} // end namespace SCIRun

#endif // HDF5DataReader_h
