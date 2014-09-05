
#ifndef BuilderImpl_h
#define BuilderImpl_h

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/ComponentImpl.h>

namespace sci_cca {

class Builder : virtual public ComponentImpl {

public:
  Builder();
  ~Builder();

  void ui();

private:
  void menu();
  void create_component();
  void connect_components();
  void read_input_script();
  void list_active_components();
  void shutdown_framework();

  // Framework specific functions:
  //virtual void setServices( const Services &);  <- using base class for now

};

} // namespace sci_cca

#endif 

