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

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Thread/Mutex.h>

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class CollectFields : public Module {
public:
  CollectFields(GuiContext*);
  virtual void execute();
  virtual void tcl_command(GuiArgs& args, void* userdata);

private:
  std::list<FieldHandle> buffer_;
  Mutex bufferlock_;
  int buffersize_;
  
  GuiInt buffersizegui_;
};


DECLARE_MAKER(CollectFields)
CollectFields::CollectFields(GuiContext* ctx)
  : Module("CollectFields", ctx, Source, "FieldsCreate", "ModelCreation"),
  buffersizegui_(ctx->subVar("buffersize")),
  bufferlock_("Lock for internal buffer of module"),
  buffersize_(0)
{
}


void CollectFields::execute()
{
  FieldHandle Input, Output;
  MatrixHandle BufferSize;
  if (!(get_input_handle("Field",Input,true))) return;
  get_input_handle("BufferSize",BufferSize,false);
  
  SCIRunAlgo::ConverterAlgo calgo(this);
  
  if (BufferSize.get_rep())
  {
    calgo.MatrixToInt(BufferSize,buffersize_);
    buffersizegui_.set(buffersize_);
    get_ctx()->reset(); 
  }
  
  buffersize_ = buffersizegui_.get();
  
  bufferlock_.lock();
  buffer_.push_back(Input);
  while (buffer_.size() > buffersize_) buffer_.pop_front();
  bufferlock_.unlock();
  
  SCIRunAlgo::FieldsAlgo algo(this);

  bufferlock_.lock();
  algo.GatherFields(buffer_,Output);
  bufferlock_.unlock();

  send_output_handle("Fields",Output,true);
}


void CollectFields::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("CollectFields needs a minor command");
    return;
  }

  if( args[1] == "reset" )
  {
    bufferlock_.lock();
    buffer_.clear();
    bufferlock_.unlock();

    return;
  }
  else 
  {
    // Relay data to the Module class
    Module::tcl_command(args, userdata);
  }
}


} // End namespace ModelCreation


