#ifndef __OPERATORS_ScalarFieldNormalize_H__
#define __OPERATORS_ScalarFieldNormalize_H__

#include "UnaryFieldOperator.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Uintah/Dataflow/Modules/Operators/UnaryFieldOperator.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using namespace SCIRun;

  class ScalarFieldNormalize: public Module, public UnaryFieldOperator {
  public:
    ScalarFieldNormalize(GuiContext* ctx);
    virtual ~ScalarFieldNormalize() {}
    
    virtual void execute(void);
    
  protected:
    template<class ScalarField>       
     void normalizeScalarField(ScalarField* input,ScalarField* output);
    
  private:
    GuiInt xIndex_, yIndex_, zIndex_;

    FieldIPort *in;
    FieldOPort *sfout;
  };
}

#endif // __OPERATORS_ScalarFieldNormalize_H__

