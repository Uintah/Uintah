#include <gfx/std.h>
#include <fstream.h>

#ifndef WIN32
#  include <unistd.h>
#  include <strstream.h>
#else
#  include <strstrea.h>
#  ifndef __CYGWIN32__
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#    define getpid() GetCurrentProcessId()
#  endif
#endif


#include <gfx/sys/futils.h>

static unsigned long tempctr = 1;

tempFilename::tempFilename(char *dir)
{
    ostrstream buf;
    buf << dir << "/tmp" << getpid() << "-" << tempctr++;
    buf << '\0';
    name = buf.str();
}

tempFilename::~tempFilename()
{
    delete name;
}



#include <stdarg.h>

#ifdef WIN32
istream *pipe_input_stream(char *cmd, ...)
{
    fatal_error("Pipe input streams not supported under Win32.");
    return NULL;
}
#else

//
// Yes, this is bogus.  But right now, I don't want to take the time
// to actually hook the stream up to a real pipe.  Why can't C++ have
// a simple analog of popen?
istream *pipe_input_stream(char *cmd, ...)
{
    va_list ap;

    ostrstream buf;
    buf << cmd;

    va_start(ap, cmd);
    for(char *part=va_arg(ap, char*); part; part=va_arg(ap, char*))
    {
	buf << " " << part;
    }
    va_end(ap);

    tempFilename tmp;
    buf << " >" << tmp.filename();

    buf << '\0';
    char *cmdline = buf.str();

    system(cmdline);
    istream *stream = new ifstream(tmp.filename());
    unlink(tmp.filename());

    delete cmdline;
    return stream;
}
#endif
