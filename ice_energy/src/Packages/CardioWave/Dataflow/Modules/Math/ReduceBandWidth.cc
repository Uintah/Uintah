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
 *  ReduceBandWidth.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Matrix.h>
#include <Packages/CardioWave/Core/Algorithms/ExtraMath.h>
#include <string>
#include <iostream>


// We may move this to a more general spot in the SCIRun tree
namespace SCIRun {

using namespace SCIRun;

class ReduceBandWidth : public Module {
public:
  ReduceBandWidth(GuiContext*);

  virtual ~ReduceBandWidth();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);

private:

  GuiString  guiinputbw_;
  GuiString  guioutputbw_;
  GuiString  guimethod_;

};


DECLARE_MAKER(ReduceBandWidth)
ReduceBandWidth::ReduceBandWidth(GuiContext* ctx)
  : Module("ReduceBandWidth", ctx, Source, "Math", "CardioWave"),
  guiinputbw_(ctx->subVar("input-bw")),
  guioutputbw_(ctx->subVar("output-bw")),
  guimethod_(ctx->subVar("method"))
{
}

ReduceBandWidth::~ReduceBandWidth(){
}

void ReduceBandWidth::execute()
{
  MatrixIPort *imatrix, *irhs;
  MatrixOPort *omatrix, *orhs;
  MatrixOPort *mapping, *imapping;
  
  if (!(imatrix = static_cast<MatrixIPort *>(get_iport("Matrix"))))
  {
    error("Could not find matrix input port");
    return;
  }

  if (!(omatrix = static_cast<MatrixOPort *>(get_oport("Matrix"))))
  {
    error("Could not find matrix output port");
    return;
  }

  if (!(irhs = static_cast<MatrixIPort *>(get_iport("RHS"))))
  {
    error("Could not find RHS input port");
    return;
  }

  if (!(orhs = static_cast<MatrixOPort *>(get_oport("RHS"))))
  {
    error("Could not find RHS output port");
    return;
  }

  if (!(mapping = static_cast<MatrixOPort *>(get_oport("Mapping"))))
  {
    error("Could not find mapping output port");
    return;
  }

  if (!(imapping = static_cast<MatrixOPort *>(get_oport("Inverse Mapping"))))
  {
    error("Could not find mapping output port");
    return;
  }

  std::string method = guimethod_.get();
  std::string special = guimethod_.get();
  
  MatrixHandle imatrixH,irhsH,omatrixH,orhsH,mappingH,imappingH;
  
  omatrixH = 0;
  orhsH = 0;
  mappingH = 0;
  imappingH = 0;
  
  imatrix->get(imatrixH);
  irhs->get(irhsH);



  if(imatrixH.get_rep() == 0)
  {
    error("No input matrix is given!");
    return;
  }
  
  ExtraMath em;
  
  int inputbw = em.computebandwidth(imatrixH);
  
  bool calcmapping = false;
  bool calcimapping = false;
  
  if (mapping->nconnections()) calcmapping = true;
  if (imapping->nconnections()) calcimapping = true;
  
  if (method == "rcm")
  {
    em.rcm(imatrixH,omatrixH,mappingH,calcmapping,imappingH,calcimapping);
  }
  if (method == "cm")
  {
    em.cm(imatrixH,omatrixH,mappingH,calcmapping,imappingH,calcimapping);
  }

  if (irhsH.get_rep() != 0)
  {
    em.mapmatrix(irhsH,orhsH,mappingH);
  }

  int outputbw = em.computebandwidth(omatrixH);

  if (omatrixH.get_rep()) omatrix->send(omatrixH);
  if (orhsH.get_rep()) orhs->send(orhsH);
  if (mappingH.get_rep()) mapping->send(mappingH); 
  if (imappingH.get_rep()) imapping->send(imappingH); 

  {
    std::ostringstream oss;
    oss << "Bandwidth input matrix = " << inputbw;
    guiinputbw_.set(oss.str());
  }
  
  {
    std::ostringstream oss;
    oss << "Bandwidth output matrix = " << outputbw;
    guioutputbw_.set(oss.str());
  }
  
  reset_vars();
}

void
 ReduceBandWidth::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


