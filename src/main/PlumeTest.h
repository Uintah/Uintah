#ifndef PlumeTest_h
#define PlumeTest_h

#include <Core/Thread/Runnable.h>
using namespace SCIRun;
using namespace sci::cca::core;

class PlumeTest : public Runnable 
{
public:
  PlumeTest( const CoreFramework::pointer &);
  virtual ~PlumeTest();
  
  virtual void run();
protected:
  CoreFramework::pointer framework;
};

#endif // PlumeTest_h
