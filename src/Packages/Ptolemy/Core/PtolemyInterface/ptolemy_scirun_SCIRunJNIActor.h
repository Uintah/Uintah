#include <jni.h>

/* Header for class ptolemy_scirun_SCIRunJNIActor */

#ifndef _Included_ptolemy_scirun_SCIRunJNIActor
#define _Included_ptolemy_scirun_SCIRunJNIActor
#ifdef __cplusplus
extern "C" {
#endif
#undef ptolemy_scirun_SCIRunJNIActor_COMPLETE
#define ptolemy_scirun_SCIRunJNIActor_COMPLETE -1L
#undef ptolemy_scirun_SCIRunJNIActor_CLASSNAME
#define ptolemy_scirun_SCIRunJNIActor_CLASSNAME 1L
#undef ptolemy_scirun_SCIRunJNIActor_FULLNAME
#define ptolemy_scirun_SCIRunJNIActor_FULLNAME 2L
#undef ptolemy_scirun_SCIRunJNIActor_LINKS
#define ptolemy_scirun_SCIRunJNIActor_LINKS 4L
#undef ptolemy_scirun_SCIRunJNIActor_CONTENTS
#define ptolemy_scirun_SCIRunJNIActor_CONTENTS 8L
#undef ptolemy_scirun_SCIRunJNIActor_DEEP
#define ptolemy_scirun_SCIRunJNIActor_DEEP 16L
#undef ptolemy_scirun_SCIRunJNIActor_ATTRIBUTES
#define ptolemy_scirun_SCIRunJNIActor_ATTRIBUTES 32L
/* Inaccessible static: _DEFAULT_WORKSPACE */
/* Inaccessible static: class_00024ptolemy_00024data_00024type_00024Typeable */
/*
 * Class:     ptolemy_scirun_SCIRunJNIActor
 * Method:    sendRecordToken
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_ptolemy_scirun_SCIRunJNIActor_sendRecordToken
  (JNIEnv *, jobject);

/*
 * Class:     ptolemy_scirun_SCIRunJNIActor
 * Method:    getScirun
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_ptolemy_scirun_SCIRunJNIActor_getScirun
  (JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
