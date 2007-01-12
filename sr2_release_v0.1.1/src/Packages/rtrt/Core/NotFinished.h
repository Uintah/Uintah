
/**
 * Utility class to provide a simple mechanism for denoting portions of
 * code that aren't finished.  This just prints out a message on a debug
 * stream that says "<msg>: Not finished <filename> (line <lineno>)"
 */

#define NOT_FINISHED(what) cerr << what << ": Not Finished " << __FILE__ << " (line " << __LINE__ << ")\n";

