#include "AsyncReply.h"
#include "Thread.h"

/*
 * Creates a single slot for some return value.  The <i>wait</i> method
 * waits for a value to be sent from another thread via the <i>reply</i>
 * method.  This is typically used to provide a simple means of returning
 * data from a server thread.  An <b>AsyncReply</b> object is created on the
 * stack, and some request is sent (usually via a <b>Mailbox</b>) to a server
 * thread.  Then the thread will block in <i>wait</i> until the server thread
 * receives the message and responds using <i>reply</i>.
 *
 * <p><b>AsyncReply</b> is a one-shot wait/reply pair - a new <b>AsyncReply</b>
 * object must be created for each reply.  Only a single thread should
 * call <i>wait</i> and a single thread shuold call <i>reply</i>.
 */

