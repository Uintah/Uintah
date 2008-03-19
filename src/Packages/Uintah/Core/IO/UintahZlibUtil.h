
#include <zlib.h>

#include <string>

namespace Uintah {

// The following functions read from a gzipped file (or even a
// non-compressed file) one token of the requested type.  A token is
// defined as a contiguous group of characters separated by white
// space (tabs, spaces, new lines, etc.)

const std::string getString( gzFile gzFp );

double            getDouble( gzFile gzFp );

int               getInt(    gzFile gzFp );

// Returns a full line... If the line is a comment it is skipped and
// the next line is returned.  If the line is empty, an empty string
// is returned.

const std::string getLine( gzFile gzFp );



} // end namespace Uintah
