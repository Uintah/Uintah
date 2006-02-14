#ifndef RTRT_EXTERNALUIINTERFACE
#define RTRT_EXTERNALUIINTERFACE 1

namespace rtrt {

class ExternalUIInterface {
public:
  ExternalUIInterface() {}
  
  // This function in no way should block.
  virtual void stop() = 0;
};

} // end namespace rtrt

#endif // ifndef RTRT_EXTERNALUIINTERFACE
