
//		    right	---------->	left
//		    0 bit	---------->	31 bit
//so->flag(32bit) : thread alloc|LOCK_GLO|LOCK_SIG|LOCK_VCS_OWN|LOCK_VCS_SHARE
	

//이걸 추가, 변경하면 cleanupLockMut()도 같이 변경
#define SBI_LOCK_SNET		1
#define SBI_LOCK_GLO		2
#define SBI_LOCK_SIG		3
#define SBI_LOCK_MATRIX		4
#define SBI_LOCK_NETWORK	5
#define SBI_LOCK_SET		6
#define SBI_LOCK_SYNC		8
#define SBI_LOCK_FROZEN		9
#define SBI_LOCK_FLUSH		10
#define SBI_LOCK_NOTIFY		11
#define SBI_LOCK_PIPEX		12
#define SBI_LOCK_OBJECT		13
#define SBI_LOCK_GC			14
#define SBI_LOCK_MTSPACE	15
#define SBI_LOCK_HASH		16
#define SBI_LOCK_SINGLE		17
//SBI - status bit index
//b - bits, bi - bit index
#define SET_SBI(b, bi) b = b | (1 << bi)
#define RESET_SBI(b, bi) b = b & ~(1 << bi)
#define GET_SBI(b, bi) ((b >> bi) & 1)
/*
#define THREAD_MASK	0x80000000
#define LOCK_PA_MASK	0x40000000
#define LOCK_IT_MASK	0x20000000
#define LOCK_PI_MASK	0x10000000
#define LOCK_FA_MASK	0x08000000
#define LOCK_FB_MASK	0x04000000

#define THREAD_UMASK	0x7fffffff
#define LOCK_PA_UMASK	0xbfffffff
#define LOCK_IT_UMASK	0xdfffffff
#define LOCK_PI_UMASK	0xefffffff
#define LOCK_FA_UMASK	0xf7ffffff
#define LOCK_FB_UMASK	0xfbffffff

#define MASK_THREAD(f)	f = f | THREAD_MASK
#define MASK_LOCK_PA(f)	f = f | LOCK_PA_MASK
#define MASK_LOCK_IT(f)	f = f | LOCK_IT_MASK
#define MASK_LOCK_PI(f)	f = f | LOCK_PI_MASK
#define MASK_LOCK_FA(f)	f = f | LOCK_FA_MASK
#define MASK_LOCK_FB(f)	f = f | LOCK_FB_MASK

#define UMASK_THREAD(f)		f = f & THREAD_UMASK
#define UMASK_LOCK_PA(f)	f = f & LOCK_PA_UMASK
#define UMASK_LOCK_IT(f)	f = f & LOCK_IT_UMASK
#define UMASK_LOCK_PI(f)	f = f & LOCK_PI_UMASK
#define UMASK_LOCK_FA(f)	f = f & LOCK_FA_UMASK
#define UMASK_LOCK_FB(f)	f = f & LOCK_FB_UMASK

#define IS_THREAD(f)	(f & THREAD_MASK)
#define IS_LOCK_PA(f)	(f & LOCK_PA_MASK)
#define IS_LOCK_IT(f)	(f & LOCK_IT_MASK)
#define IS_LOCK_PI(f)	(f & LOCK_PI_MASK)
#define IS_LOCK_FA(f)	(f & LOCK_FA_MASK)
#define IS_LOCK_FB(f)	(f & LOCK_FB_MASK)
*/

