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
 *  SprintfMatrix.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <stdio.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/String.h>
#include <Dataflow/Ports/StringPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseColMajMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>


namespace SCIRun {

using namespace SCIRun;

class SprintfMatrix : public Module {
public:
  SprintfMatrix(GuiContext*);

  virtual ~SprintfMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

private:
  
  GuiString formatstring_;
};


DECLARE_MAKER(SprintfMatrix)
SprintfMatrix::SprintfMatrix(GuiContext* ctx)
  : Module("SprintfMatrix", ctx, Source, "String", "SCIRun"),
    formatstring_(ctx->subVar("formatstring"))
{
}

SprintfMatrix::~SprintfMatrix()
{
}

void SprintfMatrix::execute()
{
  StringOPort*  oport;
  StringIPort*  format_iport;
  MatrixIPort*  matrix_iport;
  std::string   format, output;
  StringHandle  stringH;
  
  MatrixHandle currentmatrix;
  int          inputport = 1;
  int          matrixindex = 0;
  double       datavalue;
  double*      dataptr;
  bool         lastport = false;
  bool         lastdata = false;
  bool         isformat = false;
  
  std::vector<char> buffer(256);  
  
  if(!(oport = dynamic_cast<StringOPort *>(get_oport(0))))
  {
    error("Could not find string output port");
    return;
  }

  if(!(format_iport = dynamic_cast<StringIPort *>(get_iport(0))))
  {
    error("Could not find format string input port");
    return;
  }

  format = formatstring_.get();
  format_iport->get(stringH);
  if (stringH.get_rep())
  {
    format = stringH->get();
  }

  size_t i = 0;
  while(i < format.size())
  {
    if (format[i] == '%')
    {
      if (i == format.size()-1)
      {
        error("Improper format string '%' is last character");
        return;
      }
            
      if (format[i+1] == '%')
      {
          output += '%'; i += 2;
      }
      else
      {
        size_t j = i+1;
        // Just to name a few printing options
        while((j < format.size())&&(format[j] != 'd')&&(format[j] != 'e')&&(format[j] != 'g')&&(format[j] != 'c')
            &&(format[j] != 'i')&&(format[j] != 'E')&&(format[j] != 'x')&&(format[j] != 'X')&&(format[j] != 's')
            &&(format[j] != 'u')&&(format[j] != 'o')&&(format[j] != 'g')&&(format[j] != 'G')&&(format[j] != 'f')
            &&(format[j] != 'F')&&(format[j] != 'A')&&(format[j] != 'a')) j++;
    
        if (j == format.size())
        {
            error("Improper format string '%..type' clause was incomplete");
            return;
        }
              
        std::string fstr = format.substr(i,j-i+1);
        
        if ((format[j] != 's')&&(format[j] != 'S')&&(format[j] != 'C')&&(format[j] != 'c'))
        {
          isformat  = true;
          datavalue = 0.0;
          while ((currentmatrix.get_rep() == 0)&&(lastport==false))
          {
            matrix_iport = dynamic_cast<MatrixIPort *>(get_iport(inputport++));
            if (matrix_iport)
            {
              matrix_iport->get(currentmatrix);
              matrixindex = 0;
              if (currentmatrix.get_rep() != 0)
              {
                if (currentmatrix->get_data_size() == 0) currentmatrix = 0;
              }
              
              if (currentmatrix.get_rep() != 0)
              {
                // Check whether we need to transpose matrix
                // If so we transpose the whole matrix
                if (currentmatrix->is_dense_col_maj())
                {
                  currentmatrix = dynamic_cast<Matrix *>(currentmatrix->dense());
                }
              }
            }
            else
            {
              lastport = true;
              lastdata = true;
            }
          }
          
          if(currentmatrix.get_rep())
          {
            dataptr = currentmatrix->get_data_pointer();
            if (matrixindex < currentmatrix->get_data_size())
            {
              datavalue = dataptr[matrixindex++];
            }
            else
            {
              datavalue = 0.0;
            }
            if (matrixindex == currentmatrix->get_data_size()) 
            { 
              currentmatrix = 0; 
              if (inputport == (numIPorts()-1)) { lastdata = true; lastport = true; }
            }
          }
        }
        
        if ((format[j] == 's')||(format[j] == 'S')||(format[j] == 'c')||(format[j] == 'C'))
        {
          // We put the %s %S back in the string so it can be filled out lateron
          // By a different module
          output += fstr;
          i = j+1;
        }

        else if ((format[j] == 'd')||(format[j] == 'o'))
        {
          int scalar = static_cast<int>(datavalue);
          snprintf(&(buffer[0]),256,fstr.c_str(),scalar);
          output += std::string(reinterpret_cast<char *>(&(buffer[0])));
          i = j+1;
        }

        else if ((format[j] == 'i')||(format[j] == 'u')||(format[j] == 'x')||(format[j] == 'X'))
        {
          unsigned int scalar = static_cast<unsigned int>(datavalue);
          snprintf(&(buffer[0]),256,fstr.c_str(),scalar);
          output += std::string(reinterpret_cast<char *>(&(buffer[0])));
          i = j+1;                
        }
        else if ((format[j] == 'e')||(format[j] == 'E')||(format[j] == 'f')||(format[j] == 'F')||
                 (format[j] == 'g')||(format[j] == 'G')||(format[j] == 'a')||(format[j] == 'A'))
        {
          std::vector<char> buffer(256);
          snprintf(&(buffer[0]),256,fstr.c_str(),datavalue);
          output += std::string(reinterpret_cast<char *>(&(buffer[0])));
          i = j+1;   
        }
      }
    }
    else if ( format[i] == '\\')
    {
      if (i < (format.size()-1))
      {
        switch (format[i+1])
        {
          case 'n': output += '\n'; break;
          case 'b': output += '\b'; break;
          case 't': output += '\t'; break;
          case 'r': output += '\r'; break;
          case '\\': output += '\\'; break;
          case '0': output += '\0'; break;
          case 'v': output += '\v'; break;
          default:
            error("unknown escape character");
            return;
        }
        i = i+2;
      }
    }
    else
    {
      output += format[i++];
    }
    
    if ((i== format.size())&&(isformat == true)&&(lastdata == false))
    {
      i = 0;
    }
    
  }

  StringHandle handle = dynamic_cast<String *>(scinew String(output));
  oport->send(handle); 
}

void
 SprintfMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


