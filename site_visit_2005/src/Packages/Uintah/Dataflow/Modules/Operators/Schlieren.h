#ifndef __OPERATORS_Schlieren_H__
#define __OPERATORS_Schlieren_H__

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

  class Schlieren: public Module, public UnaryFieldOperator{
  public:
    Schlieren(GuiContext* ctx);
    virtual ~Schlieren() {}
    
    virtual void execute(void);
    
  protected:
    template<class ScalarField>       
     void computeSchlierenImage(ScalarField* density, ScalarField* output);
    
  private:

    GuiDouble dx_, dy_, dz_;

    FieldIPort * in_;
    FieldOPort * sfout_;
  };
  
} // end namespace Uintah

#endif // __OPERATORS_Schlieren_H__

