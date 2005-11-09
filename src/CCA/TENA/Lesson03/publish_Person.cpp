//////////////////////////////////////////////////////////////////// -*- C++ -*-
/*! \file main/publish_Person.cpp
 * \version 5.1
 *
 * \brief A simple example program that publishes
 * Lesson_03::Person objects.
 *
 * This file contains a simple example main program that "publishes" (i.e,
 * creates and updates) SDOs of type Lesson_03::Person.
 *
 * Application programmers are encouraged to copy and modify excerpts of this
 * program to suit their own needs.  The entire implementation directory
 * structure for a particular object model (top-level name of OMname_Impl) can
 * be copied into a user area to be customized.  It is recommended that an
 * "original" copy of the auto-generated implementation directory structure is
 * also maintained in the development area (e.g., OMname_Impl.orig).  The
 * original directory provides the ability to perform a source code comparison
 * and create patches to assist migration to new versions of the object model
 * or middleware.
 *
 * Publishing applications will typically be interested in the customization of
 * a few associated classes defined below:
 *
 * \sa Lesson_03::Person::BasicImpl::RemoteMethodsImpl -
 * Provides the user defined behavior for any methods associated with the SDO.
 * Users are required to insert code into this class when SDO methods are
 * defined in the object model.
 */

#include <TENA/Middleware/config.h>
#include <string>
#include <ace/OS.h>             // for ACE_OS::sleep()
#include <Lesson_03/Person/BasicImpl/PublisherImpl.h>

// Include all the implementation files for contained types that are needed to
// access the state of this SDO
#include <Lesson_03/Location/Includes.h>

namespace TM = TENA::Middleware;

void init( int argc, char *argv[], DCT::Utils::BasicConfiguration &programConfig );
void parse_args( int argc, char *argv[], DCT::Utils::BasicConfiguration &programConfig );
void updateServant( Lesson_03::Person::ServantPtr pPersonServant,  long updateCount );

int main( int argc, char * argv[] )
{
  TM::Configuration tenaMiddlewareConfiguration( argc, argv );
  DCT::Utils::BasicConfiguration programConfig( "Program Options" );
  init( argc, argv, programConfig );

  try {
    TM::RuntimePtr runtime = TM::init( tenaMiddlewareConfiguration );

    std::string executionName = programConfig["executionName"].getValue< std::string >();
    TM::ExecutionPtr execution = runtime->joinExecution( executionName );

    std::string sessionName = programConfig["sessionName"].getValue< std::string >();
    TM::SessionPtr session = execution->createSession( sessionName );

    // Create a PublicationInfoImpl helper object that can be used to provide
    // the RemoteMethodsImpl access to application variables/objects.  See the
    // Lesson_03::Person::BasicImpl::PublicationInfoImpl header file for
    // usage information.
    Lesson_03::Person::PublicationInfoPtr publicationInfo(new Lesson_03::Person::BasicImpl::PublicationInfoImpl);

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
    TM::CommunicationProperties communicationProperties =
      programConfig["bestEffort"].isSet() ? TM::BestEffort : TM::Reliable;

    // Create a Person Servant that is returned as a smart pointer
    Lesson_03::Person::ServantPtr person = servantFactory->createServantUsingDefaultFactory(communicationProperties);

    long millisecondsBetweenUpdates =  programConfig["millisecondsBetweenUpdates"].getValue< long >() ;
    long numberOfUpdates = programConfig["numberOfUpdates"].getValue< long >();

    for ( long i = 1; i <= numberOfUpdates; ++i ) {
      long const microsecondsBetweenUpdates = millisecondsBetweenUpdates * 1000;

      // Here's a way to sleep that is portable across all supported platforms
      ACE_OS::sleep( ACE_Time_Value( 0, microsecondsBetweenUpdates ) );

      updateServant( person, i );
    }

    return 0;
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

  return 1;
}

void parse_args( int argc, char *argv[], DCT::Utils::BasicConfiguration &programConfig )
{
  // Specify the sample program options
  
  programConfig.addSettings()
    ( "executionName",
      DCT::Utils::Value< std::string >(),
      "Name of the Execution to join." )
    
    ( "sessionName",
      DCT::Utils::Value< std::string >().setDefault( "the_Person_Session" ),
      "Name of the Session to create." )
    
    ("bestEffort",
     "Specify the use of BestEffort (UDP/IP multicast) for data transport." )
    
    ( "numberOfUpdates",
      DCT::Utils::Value< long >().setDefault( 30 ),
      "The number of times the Person SDO "
      "should be updated before the application terminates."
      )
    
    ( "millisecondsBetweenUpdates",
      DCT::Utils::Value< long >().setDefault( 1000 ),
      "The number of milliseconds between each update of the "
      "Person SDO."
      )
    
    ("help",
     "Display this help message and exit." )
    ;

    // Initialize the sample program options
  try {
    DCT::Utils::BasicConfiguration::KeyValueList optionList;
    DCT::Utils::parseCommandLine(
				 optionList,
				 programConfig,
				 argc,
				 argv );
    
    programConfig.initializeSettingValues( optionList );
    
    /****** Parsing of environment and file options can be added here *******/
    
    // Warn about left over arguments.  Allow 1 argument for the program name
    if ( argc > 1 )
      {
	std::cout << "WARNING: The following command line options were not "
	  "recognized:\n" << DCT::Utils::ARGV( argc, argv ) << std::endl;
      }
  }
  catch ( std::exception const & ex ) {
    std::cerr << "While initializing option values, caught exception: "
	      << ex.what() << "\nUsage: " << argv[0] << ' '
	      << programConfig.getShortUsageString()
	      << "\n\nUse \"-help\" for a more detailed usage message\n"
	      << std::endl;
    
    exit(1);
  }
}


void init( int argc, char *argv[], DCT::Utils::BasicConfiguration &programConfig )
{
  try {
    // Configuration parameters are captured in TENA::Middleware::Configuration
    // object.  These parameters can be defined through command line arguments,
    // environment variables, or a configuration file.  See the file
    // TENA/Middleware/Configuration.h for additional information.  Used
    // arguments will be consumed.
    parse_args( argc, argv, programConfig);

    // Print detailed usage message and exit if "help" is set
    if ( programConfig["help"].isSet() )
    {
      programConfig.printUsage( std::cout );
      TM::Configuration tenaMiddlewareConfiguration( argc, argv );
      tenaMiddlewareConfiguration.printUsage( std::cout );
      exit(0);
    }
  
    // This checks to make sure all options have a default or set value.  If
    // not, errorMessage will contain a message with those options that still
    // need to be set.
    long millisecondsBetweenUpdates =  programConfig["millisecondsBetweenUpdates"].getValue< long >() ;
    std::string errorMessage;
    if ( ! programConfig.validate( errorMessage ) || millisecondsBetweenUpdates<0  ) {
      std::cerr << errorMessage << "\nUsage: " << argv[0] << ' '
                << programConfig.getShortUsageString()
                << "\n\nUse \"-help\" for a more detailed usage message\n"
                << std::endl;

      exit(1);
    }
  }
  catch ( std::exception const & ex ) {
    std::cerr << "The following exception was raised:\n\t" << ex.what()
              << std::endl;
  }
  catch ( ... ) {
    std::cerr << "An unexpected exception was raised!" << std::endl;
  }
}


// This is just an example that changes ALL the servant's state
// based on the updateCount argument.  Every contained SDO, attribute,
// vector, etc. is changed in some way.
void
updateServant( Lesson_03::Person::ServantPtr person, long updateCount )
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
