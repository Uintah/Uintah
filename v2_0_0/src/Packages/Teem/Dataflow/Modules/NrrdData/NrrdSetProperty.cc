/*
 *  NrrdSetProperty: Set a property for a Nrrd
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Teem/Core/Datatypes/NrrdData.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <Core/GuiInterface/GuiVar.h>

namespace SCITeem {

using namespace SCIRun;

class NrrdSetProperty : public Module {
public:
  NrrdSetProperty(GuiContext* ctx);
  virtual ~NrrdSetProperty();
  virtual void execute();

protected:
  GuiInt    nEntries_;
  GuiString sProperty_;
  GuiString sType_;
  GuiString sValue_;
  GuiInt    iReadOnly_;

  vector< GuiString* > gProperty_;
  vector< GuiString* > gType_;
  vector< GuiString* > gValue_;
  vector< GuiString* > gReadOnly_;

  unsigned int entries_;
  string property_;
  string type_;
  string value_;
  int    readonly_;

  vector< string > properties_;
  vector< string > types_;
  vector< string > values_;
  vector< int    > readOnly_;

  int nGeneration_;
  NrrdDataHandle nHandle_;

  int error_;
};

DECLARE_MAKER(NrrdSetProperty)
NrrdSetProperty::NrrdSetProperty(GuiContext* context)
: Module("NrrdSetProperty", context, Filter,"NrrdData", "Teem"),
  nEntries_(context->subVar("num-entries")),
  sProperty_(context->subVar("property")),
  sType_(context->subVar("type")),
  sValue_(context->subVar("value")),
  iReadOnly_(context->subVar("readonly")),
  entries_(0),
  error_(-1)
{
}

NrrdSetProperty::~NrrdSetProperty()
{
}

void NrrdSetProperty::execute() {


  NrrdIPort *inrrd_port = (NrrdIPort *)get_iport("Input");

  if (!inrrd_port) {
    error("Unable to initialize iport 'Input'.");
    return;
  }
  
  NrrdDataHandle nHandle;

  // The nrrd input is required.
  if (!inrrd_port->get(nHandle) || !(nHandle.get_rep()) ) {
    error( "No handle or representation" );
    return;
  }

  bool update = false;

  // If no data or a change recreate the nrrd.
  if( !nHandle_.get_rep() ||
      nGeneration_ != nHandle->generation ) {
    nGeneration_ = nHandle->generation;
  
    // Add the current properties to the display.
    for( unsigned int ic=0; ic<nHandle->nproperties(); ic++ ) {
      int    p_int;
      float  p_float;
      string p_string;

      string name = nHandle->get_property_name( ic );
      string type("other");
      string value("Can not display");
      int readonly = 1;

      if( nHandle->get_property( name, p_int ) ) {
	type = string( "int" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_int );
	value = string( tmpStr );
	readonly = 1;

      } else if( nHandle->get_property( name, p_float ) ) {
	type = string( "float" );
	char tmpStr[128];
	sprintf( tmpStr, "%f", p_float );
	value = string( tmpStr );
	readonly = 1;

      } else if( nHandle->get_property( name, p_string ) ) {
	type = string( "string" );
	value = p_string;
	readonly = 1;
      }

      ostringstream str;
      str << id << " setEntry {";
      str << name << "} ";
      str << type << " {";
      str << value << "} ";
      str << readonly << " ";

      gui->execute(str.str().c_str());
    }

    update = true;
  }

  // Save off the defaults
  nEntries_.reset();               // Number of entries
  sProperty_.reset();              // Default Property
  sType_.reset();                  // Default Type 
  sValue_.reset();                 // Default Value 
  iReadOnly_.reset();              // Default Read Only

  if( entries_ != (unsigned int) nEntries_.get() ) {
    entries_ = nEntries_.get();              // Number of entries
    update = true;
  }

  property_ = sProperty_.get();              // Default Property
  type_ = sType_.get();                      // Default Type 
  value_ = sValue_.get();                    // Default Value 
  readonly_ = iReadOnly_.get();              // Default Read Only

  int entries = gProperty_.size();

  // Remove the old entries that are not needed.
  for( int ic=entries-1; ic>=(int)entries_; ic-- ) {
    delete( gProperty_[ic] );
    delete( gType_[ic] );
    delete( gValue_[ic] );
    delete( gReadOnly_[ic] );

    gProperty_.pop_back();
    gType_.pop_back();
    gValue_.pop_back();
    gReadOnly_.pop_back();

    properties_.pop_back();
    types_.pop_back();
    values_.pop_back();
    readOnly_.pop_back();
  }

  // Add new entries that are needed.
  for( unsigned int ic=entries; ic<entries_; ic++ ) {
    char idx[24];

    sprintf( idx, "property-%d", ic );
    gProperty_.push_back(new GuiString(ctx->subVar(idx)) );

    sprintf( idx, "type-%d", ic );
    gType_.push_back(new GuiString(ctx->subVar(idx)) );

    sprintf( idx, "value-%d", ic );
    gValue_.push_back(new GuiString(ctx->subVar(idx)) );

    sprintf( idx, "readonly-%d", ic );
    gReadOnly_.push_back(new GuiString(ctx->subVar(idx)) );

    properties_.push_back("");
    types_.push_back("");
    values_.push_back("");
    readOnly_.push_back(0);
  }


  for( unsigned int ic=0; ic<entries_; ic++ ) {
    gProperty_[ic]->reset();
    gType_[ic]->reset();
    gValue_[ic]->reset();

    string tmpStr;

    // Compare with the current stored properties.
    tmpStr = gProperty_[ic]->get();
    if( tmpStr != properties_[ic] ) {
      properties_[ic] = tmpStr;
      update = true;
    }

    tmpStr = gType_[ic]->get();
    if( tmpStr != types_[ic] ) {
      types_[ic] = tmpStr;
      update = true;
    }

    tmpStr = gValue_[ic]->get();
    if( tmpStr != values_[ic] ) {
      values_[ic] = tmpStr;
      update = true;
    }

    // Compare with the current nrrd properties.
    int    p_int;
    float  p_float;
    string p_string;

    if( types_[ic] == string( "int" ) &&
	nHandle->get_property( name, p_int ) ) {
      if( p_int != atoi( values_[ic].c_str() ) )
	update = true;

    } else if( types_[ic] == string( "float" ) &&
	nHandle->get_property( name, p_float ) ) {
      if( p_float != atof( values_[ic].c_str() ) )
	update = true;

    } else if( types_[ic] == string( "string" ) &&
	nHandle->get_property( name, p_string ) ) {
      if( p_string != values_[ic] )
	update = true;
    }
  }

  if( update == true ||
      error_ == true  ) {

    error_ = false;

    for( unsigned int ic=0; ic<entries_; ic++ ) {

      if( types_[ic] == "int" ) {
	int p_int = (int) atoi(values_[ic].c_str());
	nHandle->set_property(properties_[ic], p_int, false);

      } else if( types_[ic] == "float" ) {
	float p_float = (float) atof(values_[ic].c_str());
	nHandle->set_property(properties_[ic], p_float, false);

      } else if( types_[ic] == "string" ) {
	string p_string = (string) values_[ic];
	nHandle->set_property(properties_[ic], p_string, false);

      } else if( types_[ic] == "unknown" ) {
	error( properties_[ic] + " has an unknown type" );
	error_ = true;
	return;
      }
    }

    // Remove the deleted properties.
    for( unsigned int ic=0; ic<nHandle->nproperties(); ic++ ) {
      string name = nHandle->get_property_name( ic );

      unsigned int jc;

      for( jc=0; jc<entries_; jc++ ) {
	if( name == properties_[jc] )
	  break;
      }

      if( jc == entries_ )
	nHandle->remove_property( name );
    }

    nHandle_ = nHandle;
    nHandle_->generation++;
    nGeneration_ = nHandle->generation;
  }


  // Get a handle to the output Field port.
  if( nHandle_.get_rep() )
  {
    NrrdOPort *onrrd_port =
      (NrrdOPort *) get_oport("Output");

    if (!onrrd_port) {
      error("Unable to initialize "+name+"'s oport\n");
      return;
    }

    // Send the data downstream
    onrrd_port->send( nHandle_ );
  }
}

} // End namespace SCIRun
