/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
      m_label = nonconstVar;

      m_active = varActive;      

      setBenignValue();
      reset();
    };

    virtual ~ApplicationReductionVariable()
    {
      VarLabel::destroy(m_label);
    }
    
    void setBenignValue()
    {
      m_bool_var.setBenignValue();
      m_min_var.setBenignValue();
      m_max_var.setBenignValue();
    }

    void reset()
    {
      m_reduction = false;
      
      m_count = 0;
      m_overriddenValue = false;
    }

    // The set value call should be used before the reduction.
    void setValue( DataWarehouse * new_dw, bool val )
    {
      // Idiot proofing - If the reduction has occured do an override.
      if( m_reduction )
        overrideValue( new_dw, val );
      else {      
        m_bool_value = val;
        m_overrideValue = true;
      }
    }

    void setValue( DataWarehouse * new_dw, double val )
    {
      // Idiot proofing - If the reduction has occured do an override.
      if( m_reduction )
        overrideValue( new_dw, val );
      else {      
        m_double_value = val;
        m_overrideValue = true;
      }
    }

    // The override value call should be used after the reduction.
    void overrideValue( DataWarehouse * new_dw, bool val )
    {
      if( m_reduction ) {
        m_overriddenValue = true;
        
        new_dw->override( bool_or_vartype( val ), m_label);

        // Get the reduced value.
        new_dw->get( m_bool_var, m_label );
      }
      // Idiot proofing - If the reduction has not occured do a set.
      else
        setValue( new_dw, val );
    }

    void overrideValue( DataWarehouse * new_dw, double val )
    {
      if( m_reduction ) {
        m_overriddenValue = true;

        if( m_label->typeDescription() == min_vartype::getTypeDescription() ) {
          new_dw->override(min_vartype( val ), m_label);
          // Get the reduced value.
          new_dw->get( m_max_var, m_label );
        }
        else if( m_label->typeDescription() == max_vartype::getTypeDescription() ) {
          new_dw->override(max_vartype( val ), m_label);
          // Get the reduced value.
          new_dw->get( m_max_var, m_label );
        }
      }
      // Idiot proofing - If the reduction has not occured do a set.
      else
        setValue( new_dw, val );
    }

    void reduce( DataWarehouse * new_dw )
    {
      Patch* patch = nullptr;

      m_bool_var.setBenignValue();
      m_min_var.setBenignValue();
      m_max_var.setBenignValue();

      // Reduce only if active.
      if (m_active) {

        if( m_label->typeDescription() == bool_or_vartype::getTypeDescription() )
        {
          // If the user gave a value use it.
          if ( m_overrideValue ) {
            new_dw->put( bool_or_vartype( m_bool_value ), m_label);
            m_overrideValue = false;
            m_overriddenValue = true;
          }
          // If the value does not exists put a benign value into the
          // warehouse.
          else if (!new_dw->exists(m_label, -1, patch)) {
            new_dw->put(m_bool_var, m_label);
          }

          // Only reduce if on more than one rank
          if( Parallel::getMPISize() > 1 ) {
            new_dw->reduceMPI(m_label, 0, 0, -1);
          }

          // Get the reduced value.
          new_dw->get( m_bool_var, m_label );
        }
        else if( m_label->typeDescription() == min_vartype::getTypeDescription() )
        {
          // If the user gave a value use it.
          if ( m_overrideValue ) {
            new_dw->put(min_vartype( m_double_value ), m_label);
            m_overrideValue = false;
            m_overriddenValue = true;
          }
          // If the value does not exists put a benign value into the
          // warehouse.
          else if (!new_dw->exists(m_label, -1, patch)) {
            new_dw->put(m_min_var, m_label);
          }

          // Only reduce if on more than one rank
          if( Parallel::getMPISize() > 1 ) {
            new_dw->reduceMPI(m_label, 0, 0, -1);
          }

          // Get the reduced value.
          new_dw->get( m_min_var, m_label );
        }
        else if( m_label->typeDescription() == max_vartype::getTypeDescription() )
        {
          // If the user gave a value use it.
          if ( m_overrideValue ) {
            new_dw->put(max_vartype( m_double_value ), m_label);
            m_overrideValue = false;
          }
          // If the value does not exists put a benign value into the
          // warehouse.
          else if (!new_dw->exists(m_label, -1, patch)) {
            new_dw->put(m_max_var, m_label);
          }

          // Only reduce if on more than one rank
          if( Parallel::getMPISize() > 1 ) {
            new_dw->reduceMPI(m_label, 0, 0, -1);
          }

          // Get the reduced value.
          new_dw->get( m_max_var, m_label );
        }
      }

      m_reduction = true;
    }

    void setActive( bool val ) { m_active = val; }
    bool getActive() const { return m_active; }
    
    const VarLabel    * getLabel() const { return m_label; }
    const std::string   getName()  const { return m_label->getName(); }

    double getValue() const
    {
      if( m_active )
      {
        if( m_label->typeDescription() == bool_or_vartype::getTypeDescription() )
          return double(m_bool_var);
        else if( m_label->typeDescription() == min_vartype::getTypeDescription() )
          return m_min_var;
        else if( m_label->typeDescription() == max_vartype::getTypeDescription() )
          return m_max_var;
        else
          return 0;
      }
      else
        return 0;
    }

    bool isBenignValue() const
    {
      if( m_active )
      {
        if( m_label->typeDescription() == bool_or_vartype::getTypeDescription() )
          return m_bool_var.isBenignValue();
        else if( m_label->typeDescription() == min_vartype::getTypeDescription() )
          return m_min_var.isBenignValue();
        else if( m_label->typeDescription() == max_vartype::getTypeDescription() )
          return m_max_var.isBenignValue();
        else
          return true;
      }
      else
        return true;
    }

    unsigned int getCount() const { return m_count; }
    bool overridden() const { return m_overriddenValue; }
    
  private:
    bool m_active{false};

    const VarLabel *m_label{nullptr};

    // Flag to indicate if the reduction has occured.
    bool m_reduction{false};
    // Count the number of times the value may have been set before
    // being cleared.
    unsigned int m_count{0};

    // Because this class gets put into a map it can not be a
    // template. As such, there are multiple storage variables. The
    // user need to know which one to use. Which they should given the
    // type description.

    // Also these vars hold the value of the reduction throughout the
    // execution of the task and do not get updated until the
    // reduction occurs. This scheme makes it possible to use the
    // value while the current time step is calculating the next
    // value.
    bool_or_vartype m_bool_var;
    min_vartype m_min_var;
    max_vartype m_max_var;

    // If the user has direct access to the application they can set
    // the reduction value directly rather than setting it in the data
    // warehouse. Before the reduction this value will get put into the
    // data warehouse for them thus overidding the current value.
    bool     m_bool_value{0};
    double m_double_value{0};
    bool m_overrideValue{false};
    
    bool m_overriddenValue{false};
  };

} // End namespace Uintah
   
#endif
