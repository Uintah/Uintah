//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Graph.cc
//    Author : McKay Davis
//    Date   : Wed Aug  2 10:35:24 2006


#include <Core/Skinner/Skinner.h>
#include <Core/Skinner/Drawable.h>
#include <Core/Skinner/Variables.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/String.h>
#include <Core/Events/MatrixEvent.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>


namespace SCIRun {

class Graph : public Module {
public:
  Graph(GuiContext *);
  virtual ~Graph();
  virtual void execute();
  virtual void tcl_command(GuiArgs &, void *);
private:
  virtual void set_data();
  MatrixEvent::MatrixVector_t data_;
  Skinner::Drawables_t        graphs_;
  string                      title_;

};


DECLARE_MAKER(Graph)
  
Graph::Graph(GuiContext *gui) :
  Module("Graph", gui, Sink, "Render", "SCIRun"),
  data_(0),
  graphs_(0),
  title_("Title")
{
  (new Thread(new EventManager(), "Event Manager"))->detach();
}


Graph::~Graph()
{
}

void
Graph::execute()
{
  StringHandle string_handle;
  title_ = "Title";
  if (get_input_handle("Title", string_handle, false)) {
    title_ = string_handle->get();
  }

  data_.clear();
  port_range_type range = get_iports("Input");
  if (range.first != range.second) {
    port_map_type::iterator pi = range.first;
    while (pi != range.second) {
      MatrixIPort *iport = (MatrixIPort *)get_iport(pi++->second);
      ASSERT(iport);
      MatrixHandle matrix = 0;
      iport->get(matrix);
      if (matrix.get_rep()) {
        data_.push_back(matrix);
      }
    }
  }
  set_data();
}


void
Graph::tcl_command(GuiArgs &args, void *userdata)
{
  if (args[1] == "ui")
  {
    graphs_.push_back(Skinner::load_skin(string(sci_getenv("SCIRUN_SRCDIR"))+
                                         "/Core/Skinner/Data/Graph2D.skin"));
    set_data();
  }
  else Module::tcl_command(args, userdata);
}


void
Graph::set_data()
{
  event_handle_t set_matrix_event = new MatrixEvent(data_);
  for (unsigned int i = 0; i < graphs_.size(); ++i)
  {
    graphs_[i]->process_event(set_matrix_event);
    graphs_[i]->get_vars()->insert("Graph2D::title", title_, "string", true);
    graphs_[i]->get_vars()->insert("blah", title_, "string", true);
  }    
}
}

