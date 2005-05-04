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
 *  PrintF.cc:
 *
 *  Written by:
 *  Jeroen Stinstra
 *
 */


#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/NrrdData.h>
#include <Dataflow/Ports/NrrdPort.h> 
#include <Core/Datatypes/NrrdString.h>
#include <Core/Datatypes/NrrdScalar.h>

namespace SCIRun {

using namespace SCIRun;

class PrintF : public Module {
public:
  PrintF(GuiContext*);

  virtual ~PrintF();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
  
  private:
  GuiString     formatstring_;
  GuiString     labelstring_;
};


DECLARE_MAKER(PrintF)
PrintF::PrintF(GuiContext* ctx)
  : Module("PrintF", ctx, Source, "Tools", "CardioWave"),
  formatstring_(ctx->subVar("formatstring")),
  labelstring_(ctx->subVar("labelstring"))  
{
}

PrintF::~PrintF(){
}

void PrintF::execute()
{
 
    std::string format = formatstring_.get();
    
    std::string output;
    int inputcnt = 0;
    NrrdIPort* iport;
    NrrdDataHandle NrrdH;
    
    size_t i = 0;
    while(i < format.size())
    {
        if (format[i] == '%')
        {
            if (i == format.size()-1)
            {
                error("Improper format string");
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
                    error("Improper format string");
                    return;
                }
                
                std::string fstr = format.substr(i,j-i+1);
                
                if (!(iport = reinterpret_cast<NrrdIPort*>(get_iport(inputcnt++))))
                {
                    error("Not enough input streams for format string");
                    return;
                }
                
                iport->get(NrrdH);
                
                if (NrrdH.get_rep() == 0)
                {
                    error("No nrrd supplied at input port");
                    return;
                }
                
                if (NrrdH->nrrd == 0)
                {
                    error("Detected an empty nrrd object at input");
                    return;
                }
                
                if (NrrdH->nrrd->dim < 1)
                {
                    error("The Nrrd does not seem to contain any data");
                    return;
                }
                    
                if (NrrdH->nrrd->axis[0].size == 0)
                {
                    error("The Nrrd does not seem to contain any data");
                    return;            
                }
                
                if ((format[j] == 's')||(format[j] == 'S'))
                {
                    NrrdString s(NrrdH);
                    std::string str = s.getstring();
                    
                    if (j == i+1)
                    {
                        output += str;
                    }
                    else
                    {   
                        // there is some modifier or so
                        // This implementation if naive in assuming only
                        // a buffer of 256 bytes. This needs to be altered one
                        // day.
                        std::vector<char> buffer(256);
                        ::snprintf(&(buffer[0]),256,fstr.c_str(),str.c_str());
                        output += std::string(static_cast<char *>(&(buffer[0])));
                    }
                    i = j+1;
                }
                else if ((format[j] == 'd')||(format[j] == 'o'))
                {
                    NrrdScalar s(NrrdH);
                    int scalar = 0;
                    if(!(s.get(scalar))) error("Nrrd does not contain a scalar value");
                    // This implementation if naive in assuming only
                    // a buffer of 256 bytes. This needs to be altered one
                    // day.                  
                    std::vector<char> buffer(256);
                    ::snprintf(&(buffer[0]),256,fstr.c_str(),scalar);
                    output += std::string(reinterpret_cast<char *>(&(buffer[0])));
                    i = j+1;
                }

                else if ((format[j] == 'i')||(format[j] == 'u')||(format[j] == 'x')||(format[j] == 'X'))
                {
                    NrrdScalar s(NrrdH);
                    unsigned int scalar = 0;
                    if(!(s.get(scalar))) error("Nrrd does not contain a scalar value");
                    // This implementation if naive in assuming only
                    // a buffer of 256 bytes. This needs to be altered one
                    // day.                  
                    std::vector<char> buffer(256);
                    ::snprintf(&(buffer[0]),256,fstr.c_str(),scalar);
                    output += std::string(reinterpret_cast<char *>(&(buffer[0])));
                    i = j+1;                
                }
                else if ((format[j] == 'e')||(format[j] == 'E')||(format[j] == 'f')||(format[j] == 'F')||
                         (format[j] == 'g')||(format[j] == 'G')||(format[j] == 'a')||(format[j] == 'A'))
                {
                    NrrdScalar s(NrrdH);
                    double scalar = 0;
                    if(!(s.get(scalar))) error("Nrrd does not contain a scalar value");
                    // This implementation if naive in assuming only
                    // a buffer of 256 bytes. This needs to be altered one
                    // day.                  
                    std::vector<char> buffer(256);
                    ::snprintf(&(buffer[0]),256,fstr.c_str(),scalar);
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
    }
    
    NrrdOPort *oport;
    
	if(!(oport = static_cast<NrrdOPort *>(get_oport(0))))
	{
		error("Could not find nrrdstring output port");
		return;
	}    
    
    // Convert the string into a Nrrd object
    // The NrrdString Object does all the convertions
    NrrdString nrrdstring(output);
    NrrdDataHandle nrrdH = nrrdstring.gethandle();    
    
    // Add the name of the object
    // This can be handy for instance when using bundles
    std::string label = labelstring_.get();
    if (label != "") nrrdH->set_property("name",label,false);
    
    oport->send(nrrdH);
}

void
 PrintF::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace SCIRun


