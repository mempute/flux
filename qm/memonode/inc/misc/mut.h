#ifndef _H_MUT
#define _H_MUT

#include "misc/basys.h"
#include "misc/xt.h"
#include "misc/flag.h"

#define MUT_VAR(sbi)		sbi_mut	##sbi
#define MUTVAR(sbi)			MUT_VAR(sbi)
#define MUT_NAME(sbi)		"sbi_mut"	#sbi
#define MUTNAME(sbi)		MUT_NAME(sbi)

#ifdef OPT_WIN
//#include <winuser.h>
//#include <winbase.h>
#define DECL_EXT_MUT_(hm)			extern hmutex hm
#define DECL_MUT_(hm)				hmutex hm
#define CRE_MUT_(hm, rv)			{	hm = CreateMutex(NULL, FALSE, NULL);\
										if(hm == NULL) rv = -1;\
										else rv = 0;\
										regist_dbg_pointer(hm, "CRE_MUT_");\
									}
#define CLOSE_MUT_(hm)				{\
										CloseHandle(hm);\
										remove_dbg_pointer(hm);\
									}
#define LOCK_MUT_(hm)				WaitForSingleObject(hm, INFINITE)
#define UNLOCK_MUT_(hm)				ReleaseMutex(hm)
#define TIMEDLOCK_MUT_(hm, tmout)	WaitForSingleObject(hm, tmout)

//tmout - 밀리초
#else
#include <pthread.h>
#define DECL_EXT_MUT_(hm)	extern hmutex hm
#define DECL_MUT_(hm)		hmutex hm

#if defined(OPT_MAC) || defined(OPT_BSD) || defined(OPT_UNIX)
#define PTHREAD_MUTEX_RECURSIVE_ALLOW   PTHREAD_MUTEX_RECURSIVE
#else
#define PTHREAD_MUTEX_RECURSIVE_ALLOW   PTHREAD_MUTEX_RECURSIVE_NP
#endif

#define CRE_MUT_(hm, rv) {\
    pthread_mutexattr_t attr;\
    intx val = PTHREAD_MUTEX_RECURSIVE_ALLOW;\
    pthread_mutexattr_init(&attr);\
    /*pthread_mutexattr_setkind_np(&attr, val);kkk*/\
    pthread_mutexattr_settype(&attr, val);\
    if(pthread_mutex_init(&hm, &attr)) rv = -1;\
    else rv = 0;\
    regist_dbg_pointer(&hm, "CRE_MUT_");\
}

#define CLOSE_MUT_(hm)		{\
								pthread_mutex_destroy(&hm);\
								remove_dbg_pointer(&hm);\
							}
#define LOCK_MUT_(hm)		pthread_mutex_lock(&hm)
#define UNLOCK_MUT_(hm)		pthread_mutex_unlock(&hm)
#define TIMEDLOCK_MUT_(sbi, tmout)		{\
	struct timespec deltatime;\
	deltatime.tv_sec = 0;\
	deltatime.tv_nsec = tmout * 1000;\
	pthread_mutex_timedlock(&hm, &deltatime);\
}
#endif  

#define DECL_EXT_MUT(sbi)			DECL_EXT_MUT_(MUTVAR(sbi))
#define DECL_MUT(sbi)				DECL_MUT_(MUTVAR(sbi))
#define CRE_MUT(sbi, rv)			CRE_MUT_(MUTVAR(sbi), rv)
#define CLOSE_MUT(sbi)				CLOSE_MUT_(MUTVAR(sbi))
#define LOCK_MUT(sbi)				LOCK_MUT_(MUTVAR(sbi))		
#define UNLOCK_MUT(sbi)				UNLOCK_MUT_(MUTVAR(sbi))
#define TIMEDLOCK_MUT(sbi, tmout)	TIMEDLOCK_MUT_(MUTVAR(sbi), timout)

#ifdef DEF_SINGLE_THR
#define BeginBlock(mut)
#define EndBlock(mut)
#define CreateBlock(mut, rv)
#define ReleaseBlock(mut)

#else 
#define BeginBlock(mut)			LOCK_MUT_(mut)
#define EndBlock(mut)			UNLOCK_MUT_(mut)
#define CreateBlock(mut, rv)	CRE_MUT_(mut, rv)
#define ReleaseBlock(mut)		CLOSE_MUT_(mut)

#endif

typedef struct {
	hmutex cmutex;
	intx cntlock;
} CountingMutex;

#define CreCMutex(count_mut, rv) {\
	CreateBlock(count_mut.cmutex, rv);\
	count_mut.cntlock = 0;\
}
#define LockCMutex(count_mut) {\
	BeginBlock(count_mut.cmutex);\
	count_mut.cntlock++;\
}
#define UnLockCMutex(count_mut) {\
	count_mut.cntlock--;\
	EndBlock(count_mut.cmutex);\
}
#define CleanupLockCMutex(count_mut) {\
	for(;count_mut.cntlock > 0; count_mut.cntlock--) {\
		EndBlock(count_mut.cmutex);\
	}\
	count_mut.cntlock = 0;\
}
#define PLockCMutex(p_count_mut) {\
	BeginBlock(p_count_mut->cmutex);\
	p_count_mut->cntlock++;\
}
#define PUnLockCMutex(p_count_mut) {\
	p_count_mut->cntlock--;\
	EndBlock(p_count_mut->cmutex);\
}

//뮤택스를 등록하면 init_mutex()에도 등록해야함
#ifdef DEF_MUT
DECL_MUT(SBI_LOCK_SNET); 
#else
DECL_EXT_MUT(SBI_LOCK_SNET); 
#endif

#ifdef __cplusplus
extern "C" {
#endif		
extern intx init_mutex(void);
#ifdef __cplusplus
}
#endif

#define MUT_NUM		20

#endif	//_H_MUT
