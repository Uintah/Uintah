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
 *  FieldSetProperty: Set a property for a Field
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>

#include <Core/GuiInterface/GuiVar.h>

namespace SCIRun {

class FieldSetProperty : public Module {
public:
  FieldSetProperty(GuiContext* ctx);
  virtual ~FieldSetProperty();
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
  vector< GuiInt*    > gReadOnly_;

  unsigned int entries_;
  string property_;
  string type_;
  string value_;
  int    readonly_;

  vector< string > properties_;
  vector< string > types_;
  vector< string > values_;
  vector< int    > readOnly_;

  int fGeneration_;
  FieldHandle fHandle_;

  int error_;
};

DECLARE_MAKER(FieldSetProperty)
FieldSetProperty::FieldSetProperty(GuiContext* context)
: Module("FieldSetProperty", context, Filter, "FieldsOther", "SCIRun"),
  nEntries_(context->subVar("num-entries")),
  sProperty_(context->subVar("property")),
  sType_(context->subVar("type")),
  sValue_(context->subVar("value")),
  iReadOnly_(context->subVar("readonly")),
  entries_(0),
  error_(-1)
{
}


FieldSetProperty::~FieldSetProperty()
{
}

void
FieldSetProperty::execute()
{
  FieldIPort *inrrd_port = (FieldIPort *)get_iport("Input");
  FieldHandle fHandle;

  // The nrrd input is required.
  if (!inrrd_port->get(fHandle) || !(fHandle.get_rep()) ) {
    error( "No handle or representation" );
    return;
  }

  bool update = false;

  // If no data or a change recreate the nrrd.
  if( !fHandle_.get_rep() ||
      fGeneration_ != fHandle->generation )
  {
    fGeneration_ = fHandle->generation;
  
    // Add the current properties to the display.
    for( unsigned int ic=0; ic<fHandle->nproperties(); ic++ ) {
      bool           p_bool;
      unsigned char  p_uchar;
      char           p_char;
      unsigned short p_ushort;
      short          p_short;
      unsigned int   p_uint;
      int            p_int;
      float          p_float;
      double         p_double;
      string         p_string;

      string pname = fHandle->get_property_name( ic );
      string type("other");
      string value("Can not display");
      int readonly = 1;

      if( fHandle->get_property( pname, p_bool ) ) {
	type = string( "bool" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_bool );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_uchar ) ) {
	type = string( "unsigned char" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_uchar );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_char ) ) {
	type = string( "char" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_char );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_ushort ) ) {
	type = string( "unsigned short" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_ushort );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_short ) ) {
	type = string( "short" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_short );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_uint ) ) {
	type = string( "unsigned int" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_uint );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_int ) ) {
	type = string( "int" );
	char tmpStr[128];
	sprintf( tmpStr, "%d", p_int );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_float ) ) {
	type = string( "float" );
	char tmpStr[128];
	sprintf( tmpStr, "%f", p_float );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_double ) ) {
	type = string( "double" );
	char tmpStr[128];
	sprintf( tmpStr, "%f", p_double );
	value = string( tmpStr );
	readonly = 0;

      } else if( fHandle->get_property( pname, p_string ) ) {
	type = string( "string" );
	value = p_string;
	readonly = 0;
      }

      ostringstream str;
      str << id << " setEntry {";
      str << pname << "} ";
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

  int entries = gProperty_.size();           // # GUI vars entries

  // Remove the GUI entries that are not needed.
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

  // Add new GUI entries that are needed.
  for( unsigned int ic=entries; ic<entries_; ic++ ) {
    char idx[24];

    sprintf( idx, "property-%d", ic );
    gProperty_.push_back(new GuiString(ctx->subVar(idx)) );

    sprintf( idx, "type-%d", ic );
    gType_.push_back(new GuiString(ctx->subVar(idx)) );

    sprintf( idx, "value-%d", ic );
    gValue_.push_back(new GuiString(ctx->subVar(idx)) );

    sprintf( idx, "readonly-%d", ic );
    gReadOnly_.push_back(new GuiInt(ctx->subVar(idx)) );

    properties_.push_back("");
    types_.push_back("");
    values_.push_back("");
    readOnly_.push_back(0);
  }


  // Look through the properties to see if any have changed.
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
    double p_double;
    string p_string;

    if( types_[ic] == string( "int" ) &&
	fHandle->get_property( properties_[ic], p_int ) ) {
      if( p_int != atoi( values_[ic].c_str() ) )
	update = true;

    } else if( types_[ic] == string( "float" ) &&
	fHandle->get_property( properties_[ic], p_float ) ) {
      if( p_float != atof( values_[ic].c_str() ) )
	update = true;

    } else if( types_[ic] == string( "double" ) &&
	fHandle->get_property( properties_[ic], p_double ) ) {
      if( p_double != atof( values_[ic].c_str() ) )
	update = true;

    } else if( types_[ic] == string( "string" ) &&
	fHandle->get_property( properties_[ic], p_string ) ) {
      if( p_string != values_[ic] )
	update = true;
    }
  }


  // Something changed so update the properties.
  if( update == true ||
      error_ == true  ) {

    error_ = false;

    for( unsigned int ic=0; ic<entries_; ic++ ) {

      if( types_[ic] == "int" ) {
	int p_int = (int) atoi(values_[ic].c_str());
	fHandle->set_property(properties_[ic], p_int, false);

      } else if( types_[ic] == "float" ) {
	float p_float = (float) atof(values_[ic].c_str());
	fHandle->set_property(properties_[ic], p_float, false);

      } else if( types_[ic] == "double" ) {
	double p_double = (double) atof(values_[ic].c_str());
	fHandle->set_property(properties_[ic], p_double, false);

      } else if( types_[ic] == "string" ) {
	string p_string = (string) values_[ic];
	fHandle->set_property(properties_[ic], p_string, false);

      } else if( types_[ic] == "unknown" ) {
	error( properties_[ic] + " has an unknown type" );
	error_ = true;
	return;
      }
    }

    // Remove the deleted properties.
    for( unsigned int ic=0; ic<fHandle->nproperties(); ic++ ) {
      string pname = fHandle->get_property_name( ic );

      unsigned int jc;

      // If the property is in the entries keep it.
      for( jc=0; jc<entries_; jc++ ) {
	if( pname == properties_[jc] )
	  break;
      }

      // Otherwise if not found remove it.
      if( jc == entries_ )
	fHandle->remove_property( pname );
    }

    // Update the handles and generation numbers.
    // NOTE: This is done in place, e.g. the field is not copied.
    fHandle_ = fHandle;
    fHandle_->generation = fHandle_->compute_new_generation();
    fGeneration_ = fHandle->generation;
  }

  // Send the data downstream
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Output");
    ofield_port->send_and_dereference( fHandle_, true );
  }
}

} // End namespace SCIRun
