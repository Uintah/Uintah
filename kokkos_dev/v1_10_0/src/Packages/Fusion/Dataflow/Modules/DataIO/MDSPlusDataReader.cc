/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  MDSPlusDataReader.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   March 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <sci_defs.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Ports/FieldPort.h>

#include <Core/Datatypes/StructCurveField.h>
#include <Packages/Fusion/share/share.h>

#include <Packages/Fusion/Core/ThirdParty/mdsPlusReader.h>

namespace Fusion {

using namespace SCIRun;

class FusionSHARE MDSPlusDataReader : public Module {
public:
  MDSPlusDataReader(GuiContext *context);

  virtual ~MDSPlusDataReader();

  virtual void execute();

protected:
  GuiString sServerName_;
  GuiString sTreeName_;
  GuiString sShotNumber_;

  string server_;
  string tree_;
  int shot_;

  bool error_;

  FieldHandle pHandle_;
};


DECLARE_MAKER(MDSPlusDataReader)


MDSPlusDataReader::MDSPlusDataReader(GuiContext *context)
  : Module("MDSPlusDataReader", context, Source, "DataIO", "Fusion"),
    sServerName_(context->subVar("serverName")),
    sTreeName_(context->subVar("treeName")),
    sShotNumber_(context->subVar("shotNumber")),
    error_(-1)
{
}

MDSPlusDataReader::~MDSPlusDataReader(){
}

void MDSPlusDataReader::execute(){

#ifdef HAVE_MDSPLUS

  string server(sServerName_.get());      // MDS+ Server
  std::string tree(sTreeName_.get());     // MDS+ Tree 
  int shot = atoi( sShotNumber_.get().c_str() );  // NIMROD shot to be opened

  if( error_  == true ||
      
      server_ != server ||
      tree_   != tree   ||
      shot_   != shot ) {

    error_ = false;

    server_ = server;
    tree_   = tree;
    shot_   = shot;

    int nPhi;  // Dimension of the data.    
   
    double *phi_data = NULL;   // phi Grid data.

    StructCurveMesh *slm = NULL;
    StructCurveField<double> *pField = NULL;
    
    MDSPlusReader mds;

    int trys = 0;
    int retVal;

    while( trys < 10 && (retVal = mds.connect( server.c_str()) ) == -2 ) {
      remark( "Waiting for the connection to become free." );
      sleep( 1 );
      trys++;
    }

    /* Connect to MDSplus */
    if( retVal == -2 ) {
      error( "Connection to Mds Server " + server + " too busy ... giving up.");
      error_ = true;
      return;
    }
    else if( retVal < 0 ) {
      error( "Connecting to Mds Server " + server );
      error_ = true;
      return;
    }
    else
      remark( "Conecting to MdsPlus Server " + server );

    // Open tree
    trys = 0;

    while( trys < 10 && (retVal = mds.open( tree.c_str(), shot) ) == -2 ) {
      remark( "Waiting for the tree and shot to become free." );
      sleep( 1 );
      trys++;
    }

    if( retVal == -2 ) {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot << " too busy ... giving up.";
      error_ = true;
      return;
    }
    if( retVal < 0 ) {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      error( str.str() );
      error_ = true;
      return;
    }
    else {
      ostringstream str;
      str << "Opening " << tree << " tree and shot " << shot;
      remark( str.str() );
    }

    int dims[3];

    // Query the server for the cylindrical components of the grid.
    phi_data =  mds.grid( "PHI", dims );

    mds.disconnect();

    //  Load the fields with the mesh and data.
    if( phi_data != NULL && dims[0] ) {
      nPhi = dims[0];

      {
	ostringstream str;
	str << "Read phi data " << nPhi-1 << " slices";
	remark( str.str() );
      }

      StructCurveMesh::Node::index_type node;

      //      slm = scinew ScanLineMesh(nPhi-1,
      //			        Point(0,0,0),
      //			        Point(nPhi-2, nPhi-2, nPhi-2) );

      slm = scinew StructCurveMesh(nPhi-1);
      pField = scinew StructCurveField<double>(slm, Field::NODE);

      pHandle_ = pField;

      for( int phi=0; phi<nPhi-1; phi++ ) {
	node = phi;

	slm->set_point(  Point(phi,phi,phi), node );
	pField->set_value(phi_data[phi], node);
      }
    }
  }

  // Get a handle to the output phi field port.
  if( pHandle_.get_rep() )
  {
    FieldOPort *ofield_port =
      (FieldOPort *) get_oport("Output Phi Field");

    if (!ofield_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    ofield_port->send( pHandle_ );
  }

#else

  error( "No MDS PLUS availible." );

#endif
}


} // End namespace Fusion
