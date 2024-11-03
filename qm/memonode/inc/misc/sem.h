#ifndef _H_SEM
#define _H_SEM

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/xt.h"

#define SEM_PREFIX			"si_sem"

#define SEM_VAR(si)			si_sem	##si
#define SEMVAR(si)			SEM_VAR(si)
#define SEM_NAME(si)		"si_sem"	#si
#define SEMNAME(si)			SEM_NAME(si)

#define LSEM_NAME(si)       "lsi_sem_"  #si
    
#define SEM_MAX_COUNT	10000
//hm - system v semaphore handle, hs - window semaphore handle, isem - semaphore index
//si - signal index
#ifdef OPT_WIN
//#include <winuser.h>
//#include <winbase.h>
typedef HANDLE hsemx;
typedef HANDLE lsemx;
typedef HANDLE usemx;
//global semahpore inter process
#define DECL_EXT_SEM_(hs)				extern hsemx hs
#define DECL_SEM_(hs)					hsemx hs
#define INIT_SEM_(hm, key, sz, rv)		rv = 0
#define CRE_SEM_(hs, isem, hm, sz, ival, rv)	{	bytex name[36];\
													sprintf(name, "%s%d", SEM_PREFIX, isem);\
													hs = CreateSemaphore(NULL, ival, sz, name);\
													if(!hs) rv = -1;\
													else rv = 0;\
													regist_dbg_pointer(hs, "CRE_SEM_");\
												}
#define OPEN_SEM_(hs, isem, rv)			{	bytex name[36];\
											sprintf(name, "%s%d", SEM_PREFIX, isem);\
											hs = OpenSemaphore(SYNCHRONIZE, FALSE, name);\
											if(!hs) rv = -1;\
											else rv = 0;	}
#define CLOSE_SEM_(hs, rv)					{\
											if(CloseHandle(hs) == 0) rv = -1;\
											remove_dbg_pointer(hs);\
										}
#define REMOVE_SEM_(hm, arg, rv)
#define SIG_SEM_(hs, isem, hm, cnt)		ReleaseSemaphore(hs, cnt, NULL)
#define WAIT_SEM_(hs, isem, hm)			WaitForSingleObject(hs, INFINITE)
//local semaphore inner process
#define DECL_EXT_LSEM_(hs)		extern lsemx hs
#define DECL_LSEM_(hs)			lsemx hs
//i는 세마포어 카운트 초기값 - 이값을 0로 주면 첫번째 wait sem호출부터 블럭에 걸린다.
#define CRE_LSEM_(hs, i, ith, rv)	{	hs = CreateSemaphore(NULL, i, SEM_MAX_COUNT, NULL);\
									if(hs == nullx) rv = -1;\
									else rv = 0;\
									regist_dbg_pointer(hs, "CRE_LSEM_");\
								}
#define CLOSE_LSEM_(hs)			{\
									CloseHandle(hs);\
									remove_dbg_pointer(hs);\
								}
#define SIG_LSEM_(hs)			ReleaseSemaphore(hs, 1, NULL)
#define WAIT_LSEM_(hs)			WaitForSingleObject(hs, INFINITE)
#define TRY_WAIT_LSEM_(hs)		WaitForSingleObject(hs, 0)
#define GET_COUNT_LSEM_(hs, rv) {\
	long l;\
	ReleaseSemaphore(hs, 1, &l);\
	rv = l;\
}

#define CRE_USEM_(hs, i, ith, rv)	CRE_LSEM_(hs, i, ith, rv)
#define CLOSE_USEM_(hs)				CLOSE_LSEM_(hs)
#define SIG_USEM_(hs, cnt)			ReleaseSemaphore(hs, cnt, NULL)
#define WAIT_USEM_(hs)				WAIT_LSEM_(hs)
#define TRY_WAIT_USEM_(hs)			TRY_WAIT_LSEM_(hs)
#define GET_COUNT_USEM_(hs, rv)		GET_COUNT_LSEM_(hs, rv)

#else
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>
#include <semaphore.h>
#include <errno.h>
typedef intx hsemx;
#ifdef OPT_MAC
typedef sem_t *lsemx;
#else 
typedef sem_t lsemx;
#endif
typedef sem_t *usemx;
//global semahpore inter process
#define DECL_EXT_SEM_(hs)						extern hsemx hs
#define DECL_SEM_(hs)							hsemx hs
#define INIT_SEM_(hm, key, sz, rv)				{	hm = semget(key, sz, IPC_CREAT|0660);\
													if(hm < 0) {\
														rv = -1;\
														msgdisp("INIT_SEM_ fail errno: %d\n", errno);\
													} else rv = 0;\
													regist_dbg_pointer(hm, "INIT_SEM_");\
												}
#define CRE_SEM_(hs, isem, hm, sz, ival, rv)	{	if(ival) SIG_SEM_(hs, isem, hm, ival);\
													hs = isem;\
													rv = 0;\
												}
#define OPEN_SEM_(hs, isem, rv)					{	hs = isem; rv = 0; }
#define CLOSE_SEM_(hs, rv)		
#define REMOVE_SEM_(hm, arg, rv)					{	if(semctl(hm, 0, IPC_RMID, arg) == -1) rv = -1;\
														remove_dbg_pointer(hm);\
												}
#define SIG_SEM_(hs, isem, hm, cnt) {\
	struct sembuf sem_arg;\
	sem_arg.sem_num = isem;\
	sem_arg.sem_op = cnt;\
	sem_arg.sem_flg = SEM_UNDO;\
	semop(hm, &sem_arg, 1);\
}			
#define WAIT_SEM_(hs, isem, hm)				SIG_SEM_(hs, isem, hm, -1)
//local semaphore inner process
#define DECL_EXT_LSEM_(hs)		extern lsemx hs
#define DECL_LSEM_(hs)			lsemx hs
//i는 세마포어 카운트 초기값 - 이값을 0로 주면 첫번째 wait sem호출부터 블럭에 걸린다.
#ifdef OPT_MAC
//ith는 모든 프로그램에서 유니크해야한다. 현재 25까지 넘버링됨
#define CRE_LSEM_(hs, i, ith, rv) {\
    hs = sem_open(LSEM_NAME(ith), O_CREAT, 0777, i);\
    if(hs == SEM_FAILED) rv = -1;\
    else {\
        regist_dbg_pointer(hs, "CRE_LSEM_");\
        rv = 0;\
    }\
}
#define CLOSE_LSEM_(hs) {\
    sem_close(hs);\
    remove_dbg_pointer(hs);\
}
#define SIG_LSEM_(hs)				sem_post(hs)
#define WAIT_LSEM_(hs)				sem_wait(hs)
#define TRY_WAIT_LSEM_(hs)			sem_trywait(hs)
#define GET_COUNT_LSEM_(hs, rv)		sem_getvalue(hs, &rv)

#define CRE_USEM_(hs, i, ith, rv)	CRE_LSEM_(hs, i, ith, rv)
#define CLOSE_USEM_(hs)				CLOSE_LSEM_(hs)
#define SIG_USEM_(hs, cnt)			for(intx i = 0;i < cnt; i++) sem_post(hs)
#define WAIT_USEM_(hs)				WAIT_LSEM_(hs)
#define TRY_WAIT_USEM_(hs)			TRY_WAIT_LSEM_(hs)
#define GET_COUNT_USEM_(hs, rv)		GET_COUNT_LSEM_(hs, rv)

#else 

#define CRE_LSEM_(hs, i, ith, rv)	{\
    rv = sem_init(&hs, 0, i);\
    regist_dbg_pointer(&hs, "CRE_LSEM_");\
}
#define CLOSE_LSEM_(hs)			{\
    sem_destroy(&hs);\
    remove_dbg_pointer(&hs);\
}
#define SIG_LSEM_(hs)				sem_post(&hs)
#define WAIT_LSEM_(hs)				sem_wait(&hs)
#define TRY_WAIT_LSEM_(hs)			sem_trywait(&hs)
#define GET_COUNT_LSEM_(hs, rv)		sem_getvalue(&hs, &rv)

#define CRE_USEM_(hs, i, ith, rv)	{\
	hs = (usemx)malloc(sizeof(lsemx));\
	if(hs) {\
		rv = sem_init(hs, 0, i);\
		regist_dbg_pointer(hs, "CRE_USEM_");\
	} else rv = -1;\
}
#define CLOSE_USEM_(hs)	{\
    sem_destroy(hs);\
    remove_dbg_pointer(hs);\
	free(hs);\
}
#define SIG_USEM_(hs, cnt)			for(intx i = 0;i < cnt; i++) sem_post(hs)
#define WAIT_USEM_(hs)				sem_wait(hs)
#define TRY_WAIT_USEM_(hs)			sem_trywait(hs)
#define GET_COUNT_USEM_(hs, rv)		sem_getvalue(hs, &rv)

#endif

#endif

#define CLEANUP_LSEM_(hs) for(;TRY_WAIT_LSEM_(hs) == 0;)

typedef struct {
	intx postsig;//세마포어의 특성상 두군데서 동시에 시그널 할수없기 때문에 on, off로 처리한다.
	lsemx lsemval;
} PresentLSemaphore;
#define CrePLSem(pls, i, ith, rv) {\
	CRE_LSEM_(pls->lsemval, i, ith, rv);\
	pls->postsig = 0;\
}
//cleanup에서 시그널 표시가 됐으면 소진하기때문에 표시를 먼저해야한다. 세마포어는 임계영역
//획득에 사용하는 것이고 세마포어를 획득한 쓰레드는 하나만 존재하기때문에 카운트를 시그널 하기
//전에 증가하고 깨어난후 감소하면 동시에 두개 쓰레드가 카운트를 조작할수없기때문에 카운트의
//증감 연산은 안전하다.
#define SigPLSem(pls) {\
	SIG_LSEM_(pls->lsemval);\
}
#define WaitPLSem(pls) {\
	WAIT_LSEM_(pls->lsemval);\
}
#define CleanupPLSem(pls) {\
	CLEANUP_LSEM_(pls->lsemval);\
}
#define ClosePLSem(pls) {\
	CLOSE_LSEM_(pls->lsemval);\
}

#ifdef __cplusplus
}
#endif

#define LSEM_KEY0	0
#define LSEM_KEY1	1
#define LSEM_KEY2	2
#define LSEM_KEY3	3
#define LSEM_KEY4	4
#define LSEM_KEY5	5
#define LSEM_KEY6	6
#define LSEM_KEY7	7
#define LSEM_KEY8	8
#define LSEM_KEY9	9
#define LSEM_KEY10	10
#define LSEM_KEY11	11
#define LSEM_KEY12	12
#define LSEM_KEY13	13
#define LSEM_KEY14	14
#define LSEM_KEY15	15
#define LSEM_KEY16	16
#define LSEM_KEY17	17
#define LSEM_KEY18	18
#define LSEM_KEY19	19
#define LSEM_KEY20	20
#define LSEM_KEY21	21
#define LSEM_KEY22	22
#define LSEM_KEY23	23
#define LSEM_KEY24	24
#define LSEM_KEY25	25
#define LSEM_KEY26	26 //k232so.
#define LSEM_KEY27	27 //k266so.
#define LSEM_KEY28	28 //k267so.
#define LSEM_KEY29	29 //k360_23so.
#define LSEM_KEY30	30 //k360_26so_2.
//k360_26so_5.#define LSEM_KEY31	31 
#define LSEM_KEY32	32 //k360_26so_7.
#define LSEM_KEY33	33 //k360_33so.
#define LSEM_KEY34	34 //k369_9so.
#define LSEM_KEY35	35 //k512so.
#define LSEM_KEY36	36


//이하 정의는 mac버전에서만 의미, 다른 곳은 의미없음, 다른 곳에서 할당한는 lsemx sequence보다 휠씬 커서 중첩되지 않게 설정
#define LSEM_SEQ_SQUE_BASE	100000
#define LSEM_SEQ_QSORT_BASE	200000
#define LSEM_SEQ_JOB_BASE	300000
#define LSEM_SEQ_MPROC_BASE	400000
#define LSEM_SEQ_TMS_BASE	500000 //k360_23so.

#endif
