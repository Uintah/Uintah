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
 *  PublishPerson.h
 *
 *  Written by:
 *   Yarden Livnat
 *   SCI Institute
 *   University of Utah
 *   November 2005
 *
 */


#include <ace/OS.h>             // for ACE_OS::sleep()
#include <CCA/TENA/Lesson03/PublishPerson.h>
#include <iostream>
//#include <unistd.h>

using namespace SCIRun;
using namespace sci::cca;
using namespace sci::cca::ports;
using namespace sci::cca::tena;
using namespace TENA::Middleware;

extern "C" Component::pointer make_cca_tena_PublishPerson()
{
    return Component::pointer(new PublishPerson());
}


PublishPerson::PublishPerson()
{
}

PublishPerson::~PublishPerson()
{
}

void PublishPerson::setServices(const Services::pointer& svc)
{
  services = svc;
  
  services->addProvidesPort( new PublishPersonGoPort(this), "go", "sci.cca.ports.GoPort", TypeMap::pointer(0));
  services->registerUsesPort("tena","cca.tena.TENAService.", 0);
  
}

int PublishPerson::go()
{
  if (services.isNull()) {
    std::cerr << "services not set. go request ignored\n";
    return 1;
  }

  std::cerr << "PublishPerson ";

  try { 
    TENAService::pointer tenaService = pidl_cast<TENAService::pointer>(services.getPort("tena"));
    Execution::pointer execution = tenaService->joinExecution("tenaTest");
    services->releasePort("tena");
    
    if ( execution.isNull() ) {
      std::cerr << "could not join execution \"tenaTest\"\n";
      return;
    }
    
    ExecutionImpl *impl = dynamic_cast<ExecutionImpl *>(execution.getPointer());
    if ( !impl ) {
      std::cerr << "Execution is not ExecutionImpl ?!\n";
      return;
    }
    
    ExecutionPtr exec = impl->getExecution();
    
    std::string sessionName("lesson03");
    
    // from Lesson03
    SessionPtr session = exec->createSession( sessionName );
    
    Lesson_03::Person::PublicationInfoPtr publicationInfo = new Lesson_03::Person::BasicImpl::PublicationInfoImpl;
    
    // Create a ServantFactory to create Person objects.
    Lesson_03::Person::ServantFactoryPtr servantFactory = 
      session->createServantFactory< Lesson_03::Person::ServantTraits >(publicationInfo );
    
    // This is just a simple program that creates and updates a single SDO
    // servant.  This program demonstrates how to update all the state for
    // Lesson_03::Person servants.  In more complex applications, a single main
    // program may create multiple servants of multiple types.
    //
    // The key feature to remember is that many of objects returned to the
    // application from the middlware are "smart pointers" that maintain a
    // count of the number of "active" references to the object within the
    // process space.  When the reference count goes to zero, the object is
    // destroyed.  User applications need to "hold onto" these ServantPtrs,
    // which can be accomplished by ensuring that they don't go out of scope
    // or are stored within a container (e.g., std::list).  If necessary,
    // the user application can invoke p.reset() on the pointer to indicate
    // that the (pointed to) object is no longer is needed by the application.
    
    std::cout << "Instantiating the Lesson_03::Person::Servant." << std::endl;
    
    // Publish the pPersonServant using either Reliable (TCP/IP) or
    // BestEffort (UDP/IP multicast).
    CommunicationProperties communicationProperties = BestEffort ;
    // programConfig["bestEffort"].isSet() ? BestEffort : Reliable;
    
    // Create a Person Servant that is returned as a smart pointer
    Lesson_03::Person::ServantPtr person = servantFactory->createServantUsingDefaultFactory(communicationProperties);
    
    long millisecondsBetweenUpdates =  1000; 
    long numberOfUpdates = 10; 
    //   long millisecondsBetweenUpdates =  programConfig["millisecondsBetweenUpdates"].getValue< long >() ;
    //   long numberOfUpdates = programConfig["numberOfUpdates"].getValue< long >();
    
    for ( long i = 1; i <= numberOfUpdates; ++i ) {
      long const microsecondsBetweenUpdates = millisecondsBetweenUpdates * 1000;
      
      // Here's a way to sleep that is portable across all supported platforms
      ACE_OS::sleep( ACE_Time_Value( 0, microsecondsBetweenUpdates ) );
      
      updateServant( person, i );
    }
  }
  catch ( std::exception const & ex )
  {
    std::cerr << "The following exception was raised:\n\t" << ex.what()
              << std::endl;
  }
  catch ( ... )
  {
    std::cerr << "An unexpected exception was raised!" << std::endl;
  }

  return 0;
}


// This is just an example that changes ALL the servant's state
// based on the updateCount argument.  Every contained SDO, attribute,
// vector, etc. is changed in some way.
void
PublishPerson::updateServant( Lesson_03::Person::ServantPtr person, long updateCount )
{
  // Make a string for use in contained string attributes
  std::string updateLabel("Update # ");
  updateLabel += boost::lexical_cast< std::string >( updateCount );

  std::cout << "Updating the Lesson_03::Person::Servant's state: "
            << updateLabel << " ... " << std::flush;

  // Get the updater, that is held in auto_ptr
  std::auto_ptr< Lesson_03::Person::PublicationStateUpdater > person_updater = person->createUpdater();
  
  person_updater->set_name( updateLabel );

  // Get a Pointer to the Local Class attribute "theLocation"
  Lesson_03::Location::Pointer location = person_updater->get_theLocation();

  location->set_x( (double) updateCount );
  location->set_y( (double) updateCount );

  person_updater->set_theLocation(location);

  // Pass the updater in to the servant to modify the state atomically
  person->commitUpdater(person_updater);

  // Now that the above commitUpdater() has returned, the changes to
  // pPersonServant's state have been handed to the Middleware for
  // transmission to any interested subscribers.  That has not happened before
  // this point.

  std::cout << "Done." << std::endl;
}
