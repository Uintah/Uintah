/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_APPLICATIONREDUCTIONVARIABLE_H
#define UINTAH_HOMEBREW_APPLICATIONREDUCTIONVARIABLE_H

#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>

namespace Uintah {

  class DataWarehouse;

/**************************************

CLASS
   ApplicationReductionVariable
   
   Short description...

GENERAL INFORMATION

   ApplicationCommon.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Application Reduction Variable 

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/
  class ApplicationReductionVariable
  {
  public:
    ApplicationReductionVariable( std::string name,
                                  const TypeDescription *varType,
                                  bool varActive = false )
    {
      // Construct the label.
      VarLabel* nonconstVar = VarLabel::create(name, varType);
      nonconstVar->allowMultipleComputes();
      label = nonconstVar;

      active = varActive;      

      setBenignValue();
    };

    virtual ~ApplicationReductionVariable()
    {
      VarLabel::destroy(label);
    }
    
    void setBenignValue()
    {
      bool_var.setBenignValue();
      min_var.setBenignValue();
      max_var.setBenignValue();
    }
    
    void setValue( bool val )
    {
      bool_value = val;
      overrideValue = true;
    }

    void setValue( double val )
    {
      double_value = val;
      overrideValue = true;
    }

    void reduce( DataWarehouse * new_dw )
    {
      Patch* patch = nullptr;

      bool_var.setBenignValue();
      min_var.setBenignValue();
      max_var.setBenignValue();

      // Reduce only if active.
      if (active) {

        if( label->typeDescription() == bool_or_vartype::getTypeDescription() )
        {
          // If the user gave a value use it.
          if ( overrideValue ) {
            new_dw->put( bool_or_vartype( bool_value ), label);
            overrideValue = false;
          }
          // If the value does not exists put a benign value into the
          // warehouse.
          else if (!new_dw->exists(label, -1, patch)) {
            new_dw->put(bool_var, label);
          }

          // Only reduce if on more than one rank
          if( Parallel::getMPISize() > 1 ) {
            new_dw->reduceMPI(label, 0, 0, -1);
          }

          // Get the reduced value.
          new_dw->get( bool_var, label );
        }
        else if( label->typeDescription() == min_vartype::getTypeDescription() )
        {
          // If the user gave a value use it.
          if ( overrideValue ) {
            new_dw->put(min_vartype( double_value ), label);
            overrideValue = false;
          }
          // If the value does not exists put a benign value into the
          // warehouse.
          else if (!new_dw->exists(label, -1, patch)) {
            new_dw->put(min_var, label);
          }

          // Only reduce if on more than one rank
          if( Parallel::getMPISize() > 1 ) {
            new_dw->reduceMPI(label, 0, 0, -1);
          }

          // Get the reduced value.
          new_dw->get( min_var, label );
        }
        else if( label->typeDescription() == max_vartype::getTypeDescription() )
        {
          // If the user gave a value use it.
          if ( overrideValue ) {
            new_dw->put(max_vartype( double_value ), label);
            overrideValue = false;
          }
          // If the value does not exists put a benign value into the
          // warehouse.
          else if (!new_dw->exists(label, -1, patch)) {
            new_dw->put(max_var, label);
          }

          // Only reduce if on more than one rank
          if( Parallel::getMPISize() > 1 ) {
            new_dw->reduceMPI(label, 0, 0, -1);
          }

          // Get the reduced value.
          new_dw->get( max_var, label );
        }
      }
    }

    void setActive( bool val ) { active = val; }
    bool getActive() const { return active; }
    
    const VarLabel * getLabel() const { return label; }

    double getValue() const
    {
      if( active )
      {
	if( label->typeDescription() == bool_or_vartype::getTypeDescription() )
	  return double(bool_var);
	else if( label->typeDescription() == min_vartype::getTypeDescription() )
	  return min_var;
	else if( label->typeDescription() == max_vartype::getTypeDescription() )
	  return max_var;
	else
	  return 0;
      }
      else
	return 0;
    }

    bool isBenignValue() const
    {
      if( active )
      {
        if( label->typeDescription() == bool_or_vartype::getTypeDescription() )
          return bool_var.isBenignValue();
        else if( label->typeDescription() == min_vartype::getTypeDescription() )
          return min_var.isBenignValue();
        else if( label->typeDescription() == max_vartype::getTypeDescription() )
          return max_var.isBenignValue();
        else
          return true;
      }
      else
        return true;
    }

  private:
    bool active{false};

    const VarLabel *label{nullptr};

    // Because this class gets put into a map it can not be a
    // template. As such, there are multiple storage variables. The
    // user need to know which one to use. Which they should given the
    // type description.
    bool_or_vartype bool_var;
    min_vartype min_var;
    max_vartype max_var;

    // If the user has direct access to the application they can set
    // the reduction value directly rather than setting it in the data
    // warehouse. Before the reduction this value will get put into the
    // data wareouse for them thus overidding the current value.
    bool     bool_value{0};
    double double_value{0};
    bool overrideValue{false};
  };

} // End namespace Uintah
   
#endif
