#ifndef SCIREXCOMPOSITER_H
#define SCIREXCOMPOSITER_H



#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glu.h>



#include <Core/Thread/Mailbox.h>
#include <Core/Thread/Runnable.h>
#include <vector>

namespace SCIRun {
class Mutex;
}
namespace Kurt {
using std::vector;
using SCIRun::Runnable;
using SCIRun::Mailbox;
using SCIRun::Mutex;

class SCIRexRenderData;
class SCIRexWindow;
class SCIRexCompositer : public Runnable {
public:

  virtual void run();
  SCIRexCompositer(SCIRexRenderData *rd);
  virtual ~SCIRexCompositer();
  void add( SCIRexWindow* r);
  void doComposite();
  void kill(){ die_ = true; }
  void SetFrame(int,int,int,int);

protected:

  vector<SCIRexWindow *> renderers;
  int xmin,ymin,xmax,ymax, begin_offset, end_offset;
  SCIRexRenderData *render_data_;
  Mutex *mutex_;
  bool die_;
};

}// end namespace Kurt
#endif
