/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Interface/DWMpiHandler.h>

#include <SCICore/Exceptions/InternalError.h>

#include <Uintah/Grid/Patch.h>
#include <Uintah/Interface/DataWarehouse.h>

#include <iostream>
#include <string>
#include <mpi.h>
#include <unistd.h> // temporary for sleep

using SCICore::Exceptions::InternalError;
using std::cerr;
using std::string;

namespace Uintah {

const int DWMpiHandler::MAX_BUFFER_SIZE = 1024;
const int DWMpiHandler::DATA_REQUEST_TAG = 123321;
const int DWMpiHandler::DATA_MESSAGE_TAG = 123322;

DWMpiHandler::DWMpiHandler() : d_dw( 0 )
{
}

DWMpiHandler::~DWMpiHandler()
{
  cerr << "DWMpiHandler shutting down...\n";
}

void
DWMpiHandler::registerDW( DataWarehouseP dw )
{
  if( d_dw.get_rep() != 0 ) {
    cerr << "DWMpiHandler: Old DW generation was: " << d_dw->d_generation
	 << "\n";
  }

  d_dw = dw;

  cerr << "DWMpiHandler: New DW generation is: " << d_dw->d_generation
       << "\n";
}

void
DWMpiHandler::shutdown( int numMpiProcs )
{
  if( numMpiProcs == 1 ) {
    cerr << "Only 1 process, DWMpiHandler isn't running\n";
    return;
  }

  MpiDataRequest request;

  request.type = Quit;

  for( int procNum = 0; procNum < numMpiProcs; procNum++ )
    {
      MPI_Bsend( (void*)&request, sizeof( request ), MPI_BYTE,
		 procNum, DATA_REQUEST_TAG, MPI_COMM_WORLD );
    }
}

void
DWMpiHandler::run()
{
  cerr << "DWMpiHandler run() called\n";
  if( !d_dw.get_rep() || d_dw->d_MpiProcesses == 1 ) {

    cerr << "Only 1 MPI process, DWMpiHandler doesn't need to be running\n";

    return; // Should I throw an exception or just return and
            // let the thread die by itself?
    //throw InternalError( "DWMpiHandler should not be running "
    //			 "if there is only one MPI process." );
  }

  MPI_Status status;
  char       buffer[ MAX_BUFFER_SIZE ];
  bool       done = false;

  cerr << "DWMpiHandler beginning while loop\n";

  while( !done ) {

    cerr << "DWMpiHandler " << d_dw->d_MpiRank 
	 << " waiting for a connection\n";

    MPI_Recv( buffer, sizeof( MpiDataRequest ), MPI_BYTE, MPI_ANY_SOURCE,
	      DATA_REQUEST_TAG, MPI_COMM_WORLD, &status );

    MpiDataRequest * request = (MpiDataRequest *) buffer;

    if( request->type == Quit ) {
      cerr << "DWMpiHandler received a 'Quit' request.\n";
      done = true;
      continue;
    }

    cerr << "DWMpiHandler " << d_dw->d_MpiRank << " received this " 
	 << "request:\n";

    cerr << "from: " << request->fromMpiRank << "\n";
    cerr << "to:   " << request->toMpiRank << "\n";
    cerr << "tag:  " << request->tag << "\n";
    cerr << "var:  " << request->varName << "\n";
    cerr << "regn: " << request->patch << "\n";
    cerr << "gen:  " << request->generation << "\n\n";

    cerr << "Status is:\n";
    cerr << "Source: " << status.MPI_SOURCE << "\n";
    cerr << "Tag:    " << status.MPI_TAG << "\n";
    cerr << "Error:  " << status.MPI_ERROR << "\n";
    cerr << "Size:   " << status.size << "\n";

    if( d_dw->d_MpiRank != request->toMpiRank || 
	status.MPI_SOURCE != request->fromMpiRank ) {
      throw InternalError( "Data Notification Message sent "
			   "to/received by wrong process..." );
    }

    if( d_dw->d_generation != request->generation ) {
      throw InternalError( "Data Notification Message received for wrong "
			   "DataWarehouse generation" );
    }

    if( request->type == ReductionVar ) {
      cerr << "Received a reduction var request... need to get it from my "
	   << "database\n";
    } else if( request->type == GridVar ) {
      cerr << "Received a grid var request... need to get it from my "
	   << "database\n";
    } else {
      throw InternalError( "Do not know how to handle this type of data" );
    }

    // Look up the varName in the DW.  ??What to do if it is not there??

    // Pull data out of DataWarehouse and pack in into "buffer"

    // figure out how big the data is...
    int size = 100;

    sprintf( buffer, "this is the data you wanted" );

    cerr << "DWMpiHandler: Sending the requested info to " 
	 << status.MPI_SOURCE << "\n";
    MPI_Send( buffer, size, MPI_BYTE, status.MPI_SOURCE, DATA_MESSAGE_TAG,
	      MPI_COMM_WORLD );
    cerr << "              Sent the requested info.\n";
  } // end while

  cerr << "DWMpiHandler run() done\n";

} // end run

} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/05/30 20:19:39  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.1  2000/05/11 20:28:58  dav
// Added DWMpiHandler.cc
//
//
