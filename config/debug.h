
/*
 * Configuration file for debugging code
 *
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#ifndef SCI_AssertFlags
#define SCI_AssertFlags -DSCI_ASSERTIONS -DSCI_ASSERTION_LEVEL=3
#endif
#define SCI_CFlags SCI_DebugCFlags SCI_OtherCFlags SCI_AssertFlags
#define SCI_CppFlags SCI_DebugCppFlags SCI_OtherCppFlags SCI_AssertFlags
