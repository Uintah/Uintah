#ifndef SCIRun_RescaleColorMap_H
#define SCIRun_RescaleColorMap_H
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
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>

namespace SCIRun {

class RescaleColorMap : public Module {
public:

  //! Constructor taking [in] id as an identifier
  RescaleColorMap(GuiContext* ctx);

  virtual ~RescaleColorMap();
  virtual void execute();

protected:
  bool success_;

private:
  GuiString gFrame_;
  GuiInt gIsFixed_;
  GuiDouble gMin_;
  GuiDouble gMax_;
  GuiInt gMakeSymmetric_;

  int isFixed_;
  double min_;
  double max_;
  int makeSymmetric_;

  ColorMapHandle cHandle_;

  int cGeneration_;

  std::vector<int> fGeneration_;

  bool error_;

  pair<double,double> minmax_;
};

} // End namespace SCIRun

#endif
