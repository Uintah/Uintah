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
 *  SubscribePerson.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   November 2005
 *
 */


#include <Core/CCA/spec/sci_sidl.h>

#include <TENA/Middleware/config.h>
#include <CCA/TENA/Lesson03/SubscribePerson.h>
#include <Lesson_03/Person/BasicImpl/SubscriberImpl.h>
#include <Lesson_03/Location/Includes.h>

#include <CCA/TENA/TENAService/ExecutionImpl.h>
#include <iostream>
//#include <unistd.h>

using namespace SCIRun;
using namespace sci::cca;
using namespace sci::cca::ports;
using namespace sci::cca::tena;
using namespace TENA::Middleware;

extern "C" Component::pointer make_cca_tena_SubscribePerson()
{
    return Component::pointer(new SubscribePerson());
}

SubscribePerson::SubscribePerson()
  : thread(0)
{
}

SubscribePerson::~SubscribePerson()
{
}

void SubscribePerson::setServices(const Services::pointer& svc)
{
  services = svc;
  
  services->registerUsesPort("tena","cca.tena.TENAService.", 0);
  
  std::cerr << "PublishPerson ";
  thread = new Thread( this, "SubscribeThread");
}

void SubscribePerson::run()
{
  try { 
    TENAService::pointer tenaService = pidl_cast<TENAService::pointer>(services->getPort("tena"));
    TENAExecution::pointer execution = tenaService->joinExecution("tenaTest");
    services->releasePort("tena");
    
    if ( execution.isNull() ) {
      std::cerr << "SubscribePerson: could not join execution \"tenaTest\"\n";
      return;
    }
    
    ExecutionImpl *impl = dynamic_cast<ExecutionImpl *>(execution.getPointer());
    if ( !impl ) {
      std::cerr << "SubscribePerson: Execution is not ExecutionImpl ?!\n";
      return;
    }
    
    ExecutionPtr exec = impl->getExecution();
    
    std::string sessionName("Lesson03_Subscribe");
    
    // from Lesson03
    SessionPtr session = exec->createSession( sessionName );
    
    unsigned int verbosity = 1;
    
    Lesson_03::Person::BasicImpl::CallbackInfoPtr  pCallbackInfo(
        new Lesson_03::Person::BasicImpl::CallbackInfo( std::cout, verbosity ) );
    
    // Create a pSubscriptonInfo object used as a mechanism to pass custom
    // variables into contained local classes.  In most circumstances, the
    // auto-generated basic implementation version is sufficient.
    Lesson_03::Person::SubscriptionInfoPtr pSubscriptionInfo(
      new Lesson_03::Person::BasicImpl::SubscriptionInfoImpl( pCallbackInfo ) );

    // Declare the application's interest in Person objects.
    session->subscribeToSDO< Lesson_03::Person::ProxyTraits >( pSubscriptionInfo );
    
    std::cout << "Subscriber ready ..." << std::endl;
    
    // This subscriber application will run until SDOs are discovered, and will
    // exit after iterationsOfDelayBeforeExit when all discovered SDOs are
    // destroyed.
    long iterationsOfDelayBeforeExit = 10;
    bool anSDOhasBeenDiscovered = false;
    long delay = iterationsOfDelayBeforeExit;
    do {
      unsigned long millisecondsPerCallbackIteration = 1000;
      unsigned long const callbackIntervalInUsec = millisecondsPerCallbackIteration * 1000;
      
      // Callbacks are only processed during evoke methods, as shown here.
      session->evokeMultipleCallbacks( callbackIntervalInUsec );
      
      if ( 0 < pCallbackInfo->getDiscoveredSDOlist().size() ) {
	delay = iterationsOfDelayBeforeExit;
	anSDOhasBeenDiscovered = true;
      }
      else if ( anSDOhasBeenDiscovered )
        --delay;
    }
    while ( 0 < delay );
    
    return;
  }
  catch ( std::exception const & ex ) {
    std::cerr << "The following exception was raised:\n\t" << ex.what() << std::endl;
  }
  catch ( ... )
    {
      std::cerr << "An unexpected exception was raised!" << std::endl;
    }
  
  return;
}
