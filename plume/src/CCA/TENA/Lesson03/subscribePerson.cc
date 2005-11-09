//////////////////////////////////////////////////////////////////// -*- C++ -*-
/*! \file main/subscribe_Person.cpp
 * \version 5.1
 *
 * \brief A simple example program that subscribes to
 * Lesson_03::Person objects.
 *
 * This file contains a simple example program that subscribes to
 * Lesson_03::Person objects.
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
 * Subscribing applications will typically be interested in the customization of
 * a few associated classes defined below:
 *
 * \sa Lesson_03::Person::BasicImpl::CallbackInfo -
 * Can be used as a helper class that can hold application variables/objects
 * that can be passed into the callback classes.
 *
 * \sa Lesson_03::Person::BasicImpl::DiscoveryCallbackImpl -
 * Defines the application behavior necessary when a Person SDO
 * is discovered.
 *
 * \sa Lesson_03::Person::BasicImpl::StateChangeCallbackImpl -
 * Defines the application behavior necessary when a Person SDO
 * that was previously discovered is updated.
 *
 * \sa Lesson_03::Person::BasicImpl::DestructionCallbackImpl -
 * Defines the application behavior necessary when a Person SDO
 * that was previously discovered is destructed.
 */

#include <TENA/Middleware/config.h>
#include <Lesson_03/Person/BasicImpl/SubscriberImpl.h>

// Include all the implementation files for contained types that are needed to
// access the state of this SDO
#include <Lesson_03/Location/Includes.h>

int
main(
  int argc,
  char * argv[] )
{
  try
  {
    // Specify the sample program options
    DCT::Utils::BasicConfiguration programConfig( "Program Options" );

    programConfig.addSettings()
     ( "executionName",
       DCT::Utils::Value< std::string >(),
       "Name of the Execution to join." )

     ( "sessionName",
       DCT::Utils::Value< std::string >().setDefault( "the_Person_Session" ),
       "Name of the Session to create." )

     ( "verbosity",
       DCT::Utils::Value< unsigned int >().setDefault( 1 ),
       "A verbosity level argument.  A value of \"0\" is used to eliminate the "
       "output from callbacks that print out received state updates.  Any other"
       " value leaves the output on."
     )

     ( "millisecondsPerCallbackIteration",
       DCT::Utils::Value< long >().setDefault( 1000 ),
       "The number of milliseconds to spend handling callbacks each loop "
        "iteration."
     )

     ( "iterationsOfDelayBeforeExit",
       DCT::Utils::Value< long >().setDefault( 5 ),
       "The number of callback loop iterations to delay before terminating the "
       "application after all discovered SDOs have been destructed.  This "
       "creates a grace period in which new SDOs can be discovered before the "
       "application gets tired of waiting and terminates."
     )

     ("help",
      "Display this help message and exit." )
     ;

    // Configuration parameters are captured in TENA::Middleware::Configuration
    // object.  These parameters can be defined through command line arguments,
    // environment variables, or a configuration file.  See the file
    // TENA/Middleware/Configuration.h for additional information.  Used
    // arguments will be consumed.
    TENA::Middleware::Configuration tenaMiddlewareConfiguration(
      argc,
      argv );

    // Initialize the sample program options
    try
    {
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
    catch ( std::exception const & ex )
    {
      std::cerr << "While initializing option values, caught exception: "
                << ex.what() << "\nUsage: " << argv[0] << ' '
                << programConfig.getShortUsageString()
                << "\n\nUse \"-help\" for a more detailed usage message\n"
                << std::endl;

      return 1;
    }

    // Print detailed usage message and exit "help" is set
    if ( programConfig["help"].isSet() )
    {
      programConfig.printUsage( std::cout );
      tenaMiddlewareConfiguration.printUsage( std::cout );
      return 0;
    }

    long millisecondsPerCallbackIteration(
      programConfig["millisecondsPerCallbackIteration"].getValue< long >() );
    long iterationsOfDelayBeforeExit(
      programConfig["iterationsOfDelayBeforeExit"].getValue< long >() );

    // This checks to make sure all options have a default or set value.  If
    // not, errorMessage will contain a message with those options that still
    // need to be set.
    std::string errorMessage;
    if (    ( ! programConfig.validate( errorMessage ) )
         || ( 0 > millisecondsPerCallbackIteration )
         || ( 0 > iterationsOfDelayBeforeExit ) )
    {
      std::cerr << errorMessage << "\nUsage: " << argv[0] << ' '
                << programConfig.getShortUsageString()
                << "\n\nUse \"-help\" for a more detailed usage message\n"
                << std::endl;

      return 1;
    }

    // Initialize the middleware using the TENA::Middleware::Configuration
    // object.  Hold onto the RuntimePtr to enable this process to communicate
    // with one or more Executions.  Release it when this process no longer
    // needs to communicate with any Executions.
    TENA::Middleware::RuntimePtr pTENAmiddlewareRuntime(
      TENA::Middleware::init( tenaMiddlewareConfiguration ) );

    std::string executionName(
      programConfig["executionName"].getValue< std::string >() );

    // Join the Execution.  Hold onto the ExecutionPtr to indicate participation
    // in the Execution.  Let go of it to leave the Execution.
    TENA::Middleware::ExecutionPtr pExecution(
      pTENAmiddlewareRuntime->joinExecution( executionName ) );

    std::string sessionName(
      programConfig["sessionName"].getValue< std::string >() );

    // Create a Session in the Execution.  Any name for the Session may be used,
    // but a unique name within this application is required if multiple
    // application sessions exist for the same execution.
    TENA::Middleware::SessionPtr pSession(
      pExecution->createSession( sessionName ) );

    // Keeping a ProxyPtr to a discovered SDO is how an application signal it's
    // continued "interest" in the particular SDO.  If no ProxyPtr is held, then
    // the SDO Proxy will be destroyed and the application will no longer
    // receive updates for the particular SDO.  One common means to keep
    // discovered ProxyPtr's around is for the subscribing application to put
    // them in a std::list.

    // Any and all actions performed when an SDO is discovered, changed, or
    // destroyed is implemented in the Callback execute() method.  The user-
    // defined CallbackInfo provides an easy means for an application pass
    // information to and from the StateChangeCallback.
    // The generated CallbackInfo class may be modified to pass any
    // desired state to the Callbacks.

    // Create a CallbackInfo object which can be used to hold custom variables
    // that are passed into the discovery, state change, and destruction
    // callbacks.
    unsigned int verbosity(
      programConfig["verbosity"].getValue< unsigned int >() );

    Lesson_03::Person::BasicImpl::CallbackInfoPtr  pCallbackInfo(
        new Lesson_03::Person::BasicImpl::CallbackInfo( std::cout, verbosity ) );

    // Create a pSubscriptonInfo object used as a mechanism to pass custom
    // variables into contained local classes.  In most circumstances, the
    // auto-generated basic implementation version is sufficient.
    Lesson_03::Person::SubscriptionInfoPtr pSubscriptionInfo(
      new Lesson_03::Person::BasicImpl::SubscriptionInfoImpl(
        pCallbackInfo ) );

    // Declare the application's interest in Person objects.
    pSession->subscribeToSDO< Lesson_03::Person::ProxyTraits >(
      pSubscriptionInfo );

    std::cout << "Subscriber ready ..." << std::endl;

    // This subscriber application will run until SDOs are discovered, and will
    // exit after iterationsOfDelayBeforeExit when all discovered SDOs are
    // destroyed.
    bool anSDOhasBeenDiscovered( false );
    long delay( iterationsOfDelayBeforeExit );
    do
    {
      unsigned long const callbackIntervalInUsec(
        ( unsigned long )( millisecondsPerCallbackIteration ) * 1000 );

      // Callbacks are only processed during evoke methods, as shown here.
      pSession->evokeMultipleCallbacks( callbackIntervalInUsec );

      if ( 0 < pCallbackInfo->getDiscoveredSDOlist().size() )
      {
        delay = iterationsOfDelayBeforeExit;
        anSDOhasBeenDiscovered = true;
      }
      else if ( anSDOhasBeenDiscovered )
      {
        --delay;
      }
    }
    while ( 0 < delay );

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
