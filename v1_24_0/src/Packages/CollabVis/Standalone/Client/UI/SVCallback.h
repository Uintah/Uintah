#ifndef __sv_callback_h_
#define __sv_callback_h_

namespace SemotusVisum {

class GuiArgs;
class MouseEvent;

class SVCallback {
public:
  SVCallback() {}
  ~SVCallback() {}
  
  void tcl_command(GuiArgs &args, void * data);
  
protected:
  MouseEvent mouseEvent( GuiArgs &args );
};

}
#endif
