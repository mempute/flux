#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#include "ccpe.h"

extern thrret ThrRutra(thrarg arg);
extern thrret dbg_ThrRutra(thrarg arg);

class RunTrack;
class Tracker;

using namespace memput::mp;

class Begining {
public:
	void *operator new(size_t size, RunTrack *rt);
};

class SignalR : public Begining {
public:
	sytex onWait;
	sytex afterCopy;
	intx cntPrs, cntReq, idSigr, cntSig;
	hmutex srMut;
	lsemx srSem;
	SignalR *ptrLeft, *ptrRight, *ptrLeft2, *ptrRight2;
	Tracker *lreqTrk;
	RunTrack *rutra;
	
	SignalR(intx ith, RunTrack *rt)
	{
		intx rv;
		
		idSigr = ith;
		CRE_LSEM_(srSem, 0, ith, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre sem fail\n");

		CRE_MUT_(srMut, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre mut fail\n");

		rutra = rt;
		cntReq = cntPrs = cntSig = 0;
	}
	void srClose(void)
	{
		CLOSE_MUT_(srMut);
		CLOSE_LSEM_(srSem);
	}
	void srReset(void)
	{
		if(cntReq != cntPrs || cntSig > 1) {//mmm
			exit(1);
		}
		onWait = 0;
		cntReq = cntPrs = cntSig = 0;
		lreqTrk = nullx;
		afterCopy = 0;//1이면 gpu 실행, 2이면 cpu실행.
	}
	void srReset2(void)
	{
		if(cntReq != cntPrs || cntSig > 1) {//mmm
			exit(1);
		}
		onWait = 0;
		cntReq = cntPrs = cntSig = 0;
		afterCopy = 0;//1이면 gpu 실행, 2이면 cpu실행.
	}
	void checkSide(bool gpu)
	{
		afterCopy = (gpu ? 1 : 2);
	}
	void waitSign(void)
	{
		WAIT_LSEM_(srSem);
	}
	void sigSign(void)
	{
		SIG_LSEM_(srSem);
	}
	void setCount(intt c)
	{
		cntReq = c;
	}
	void srReturn(void)
	{
		LOCK_MUT_(srMut);
		cntPrs++;
		if(onWait && cntReq == cntPrs) {
			cntSig++;
			SIG_LSEM_(srSem);//조합 패스 생성 요청즉에서 대기상태이고 요청 건수를 모두 처리했으면 시그널
		}
		UNLOCK_MUT_(srMut);
	}
	void srWait(void)
	{
		LOCK_MUT_(srMut);
		onWait = 1;
		if(cntPrs < cntReq) {//요청 건수를 모두 처리할때까지 대기
			UNLOCK_MUT_(srMut);
			WAIT_LSEM_(srSem);
		} else UNLOCK_MUT_(srMut);//요청 건수를 모두 처리했으면 대기않음.
	}
	template<typename CT>
	void srPut(void)
	{
		rutra->srPut<CT>(this);//rutra의 srPut이름이 signalR의 그것과 틀리면 컴파일 에러 발생, 포팅할때 문제되면 rutra로 직접 호출
	}
};
extern void CudaDevSet(intt gid);
extern size_t getmfreegpu(intt gid);
extern void cudaerror(intt ecd, const bytet *emsg);
extern intt dbg_counting(void);
class Trace;
#define _ALIGN_SZ(sz, align) ((sz/align) * align + align)
#define ALIGN_SZ(sz, align) (sz % align ? _ALIGN_SZ(sz, align) : sz)
#define ALIGN_UNITT(sz)	ALIGN_SZ(sz, sizeof(unitt))
#define CXALC_SIZE	0x10000		//65536, 65k
class TContext {
public:
	intt *mCxtHost, *mCxtDevice;
	intt cxalcoff, ground_did;
	Trace *tcxttrc;

	intt *initTCxt(intt *mhost, intt gid)
	{
		ground_did = gid;
		cxalcoff = 0;
		if(mhost) mCxtHost = mhost;
		else mCxtHost = (intt *)malloc(CXALC_SIZE);

		cudaError_t error = cudaMalloc((void**)&mCxtDevice, CXALC_SIZE);
		if(error != cudaSuccess) throwFault(-1, "t context cuda malloc error\n");

		cudaDeviceSynchronize();
		return mCxtHost;
	}
	intt *cxalloc(intt sz, intt &off)
	{
		bytet *p = (bytet *)mCxtHost + cxalcoff;
		off = cxalcoff;
		cxalcoff += sz;
		if(cxalcoff > CXALC_SIZE) throwFault(-1, "size over cxalloc\n");
		return (intt *)p;
	}
	void cxalign(void)
	{
		cxalcoff = ALIGN_UNITT(cxalcoff);
	}
	void cxbegin(void)
	{
		cxalcoff = 0;
	}
	void syncCxt2Dev(bool gpu)
	{
		if(gpu == 0) return;
		//LOCK_MUT_(rsc::mutgpu);
		cudaError_t error = cudaMemcpy(mCxtDevice, mCxtHost, cxalcoff, cudaMemcpyHostToDevice);
		//UNLOCK_MUT_(rsc::mutgpu);
		if(error != cudaSuccess) throwFault(-1, "context cuda memcpy error\n");
	}
};

class TCxtArray {
public:
	intt *mhost;
	intt num_gpudev, focusDevGThr;
	TContext **gtrcCxt;
	void initTCxtArray(intt ngpu)
	{
		mhost = nullptr;
		num_gpudev = ngpu;
		focusDevGThr = -1;
		gtrcCxt = (TContext **)malloc(sizeof(TContext *) * ngpu);
		memset(gtrcCxt, 0x00, sizeof(TContext *) * ngpu);
	}
	void rmTCxtArray(void)
	{
		TContext *tcxt;

		for(intt i = 0; i < num_gpudev; i++) {
			tcxt = *(gtrcCxt + i);
			if(tcxt) {
				CudaDevSet(i);//메모리 해제는 생성 디바이스에 포커스하여 수행한다. 트레이스 혹은 수행
				cudaFree(tcxt->mCxtDevice);//쓰레드 종료(전체 종료)시점에 수행되므로 포커스 상관 안한다.
				cudaDeviceSynchronize();
				free(tcxt);
			}
		}
		free(mhost);
		free(gtrcCxt);
	}
	TContext *getgputxt(intt gid, Trace *trc)
	{
		if(gid < 0) gid = 0;
		else if(gid >= num_gpudev) throwFault(-1, "set gpu wrong gid\n");

		if(*(gtrcCxt + gid) == nullptr) {//gid로 set gpu device가 실행된 상태에서 호출되어야 한다.
			*(gtrcCxt + gid) = (TContext *)malloc(sizeof(TContext));
			mhost = (*(gtrcCxt + gid))->initTCxt(mhost, gid);//gpu context memory를 할당한다.
		}
		(*(gtrcCxt + gid))->tcxttrc = trc;
		return *(gtrcCxt + gid);
	}
};
class GPUHandle : public Begining {
public:
	intt gidCuda;
	GPUHandle *ptrLeft, *ptrRight, *ptrLeft2, *ptrRight2;
	virtual void hRemove(void) = 0;
};
class CudaHandle : public GPUHandle {
public:
	cublasHandle_t hCuda;

	CudaHandle(intt gid)
	{
		gidCuda = gid;
		cublasCreate(&hCuda);
	}
	void hRemove(void)
	{
		CudaDevSet(gidCuda);
		cublasDestroy(hCuda);
	}
};
class Tracker : public Begining {
public:
	bool rtShutdown;
	bool firstClass;
	intt trkType, gidTrc;
	SignalR *srGate;
	RunTrack *rutra;
	CudaHandle *execGpu;
	Tracker *ptrLeft, *ptrRight, *ptrLeft2, *ptrRight2;//2는 lreqTrk에 리스팅

	Tracker(RunTrack *rt)
	{
		rutra = rt;
		rtShutdown = false;
	}
	void trkReset(void)
	{
	}
	void gpuSetting(intt gid, CudaHandle *ghandle)
	{
		gidTrc = gid;
		execGpu = ghandle;
	}
	void onSingle(bool on);
	void ontrack(SignalR *sr, intt it = 0, bool first_class = false);
	virtual void tracking(TCxtArray *tcxt_arr) = 0;
};

struct RThreadArg {
	RunTrack *ta_rt;
	intt track_id;
};
#define RT_BASE	700000
#define SEM_RANAGE	10000
#define SRUTRA_SBASE	0 * SEM_RANAGE
#define BRUTRA_SBASE	1 * SEM_RANAGE
#define PRUTRA_SBASE	2 * SEM_RANAGE
#define CYDROME_SBASE	3 * SEM_RANAGE //(2)
//#define NEXT_SBASE		5 * SEM_RANAGE
class TrackTh {
public:
	lsemx semrt;
	Tracker *trkReq;
	intt wntrt, rntrt;
	TrackTh(intt sid)
	{
		intt rv;

		CRE_LSEM_(semrt, 0, sid, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre sem fail\n");
		}
		trkReq = nullptr;
		wntrt = rntrt = 0;
	}
	~TrackTh()
	{
		CLOSE_LSEM_(semrt);
	}
};

#define RT_DBG

#define PDG	0//1 //gpu
#define PDC	-1 //cpu
#define PDC2 -2 //only cpu
#define PDC3 -3 //must cpu
#define PDN	0
class RunTrack {
public:
	QueryContext mrtAllocator, *qcmrt;
	hmutex mutrt, mutrt_dbg;
	TrackTh *tracks[10];
	SignalR *srPool, *srLink;
	intt semBase, isrAlc, nsrAlc, nTrack;
	intt nDistri, ncreGpu, numGpu;
	longx nreqGpu, nreqCpu;
	floatt gpuMultiple;
	bool singleThr, toggleGpu, tcxtThread;
	GPUHandle **gHandle, *linkGpu;

	void addTracks(void)
	{
		++isrAlc;
		tracks[nTrack++] = new TrackTh(RT_BASE + semBase + isrAlc);
	}
	RunTrack(intx sem_base, intx nallocable, intt num_gpu)
	{
		intx rv;

		nTrack = 0;
		isrAlc = 0;
		nsrAlc = nallocable;
		semBase = sem_base;
		qcmrt = &mrtAllocator;
		InitSelPage(qcmrt);

		CRE_MUT_(mutrt, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre mut fail");
		}
#ifdef RT_DBG
		CRE_MUT_(mutrt_dbg, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre mut fail");
		}
#endif
		addTracks();
		srPool = srLink = nullptr;
		singleThr = false;
		nreqGpu = nreqCpu = 0;
		gpuMultiple = 0.5;
		numGpu = num_gpu;
		ncreGpu = 0;
		toggleGpu = true;
		tcxtThread = false;
		linkGpu = nullptr;
		if(numGpu) {
			gHandle = (GPUHandle **)rutralloc(sizeof(GPUHandle *) * numGpu);
			memset(gHandle, 0x00, sizeof(GPUHandle *) * numGpu);
		} else gHandle = nullx;
	}
	~RunTrack()
	{
		CLOSE_MUT_(mutrt);

		for(intt i = 0;i < nTrack; i++) delete tracks[i];

		for(;srLink; srLink = srLink->ptrRight2) srLink->srClose();

		ReleaseSelPage(qcmrt);

		if(linkGpu) {
			for(;linkGpu; linkGpu = linkGpu->ptrRight2) linkGpu->hRemove();
		}
	}
	void *rutralloc(size_t size)
	{
		bytex *rp;

		SelAlloc(qcmrt, size, rp);

		return rp;
	}
	template<typename CT>
	CT *gethGpu(sytet gpu, intt gid)
	{
		GPUHandle *hgpu;

		if(gpu == 1) return (CT *)1;
		//LOCK_MUT_(mutrt);
		if(gHandle && *(gHandle + gid)) {
			GET_LIST((*(gHandle + gid)), hgpu);
		} else {
			hgpu = new(this)CT(gid);
			APPEND_LIST2(linkGpu, hgpu);
			ncreGpu++;
		}
		//UNLOCK_MUT_(mutrt);
		return (CT *)hgpu;
	}
	void puthGpu(GPUHandle *hgpu, intt rsz)
	{
		LOCK_MUT_(mutrt);
		if(hgpu) {
			if(hgpu != (GPUHandle *)1) {
				APPEND_LIST((*(gHandle + hgpu->gidCuda)), hgpu);
			}
			nreqGpu -= rsz;
		} else nreqCpu -= rsz;
		UNLOCK_MUT_(mutrt);
	}
	bool policyTrack(intt rsz, bool feat_reduce, intt feat_sz, intt div, intt &width, intt &n, sytet cpm_mode);
	bool policyTrack2(intt rsz, intt &width, intt &n, sytet cpm_mode);
	bool policyTrack3(intt rsz, intt &width, intt &n, bool mean, sytet cpm_mode);
	template<typename CT> Tracker *trkGet(TContext *tcxt, intx iori, intt width, sytet gpu_exec);
	template<typename CT> Tracker *trkGet2(void);
	template<typename CT> void trkPut(Tracker *trk, bool lock);
	
	SignalR *srGet(void)
	{
		SignalR *sr;

		LOCK_MUT_(mutrt);
		if(srPool) {
			GET_LIST(srPool, sr);//pool list
			UNLOCK_MUT_(mutrt);
			sr->srReset();
			return sr;
		}
		if(++isrAlc > nsrAlc) throwFault(-1, "overflow sr\n");
		sr = new(this)SignalR(RT_BASE + semBase + isrAlc, this);
		APPEND_LIST2(srLink, sr);//해재용 보관 리스트
		UNLOCK_MUT_(mutrt);
		sr->srReset();

		return sr;
	}
	SignalR *srGet2(void) //mmm
	{
		SignalR *sr;

		LOCK_MUT_(mutrt);
		
		if(++isrAlc > nsrAlc) throwFault(-1, "overflow sr\n");
		sr = new(this)SignalR(RT_BASE + semBase + isrAlc, this);
		APPEND_LIST2(srLink, sr);//해재용 보관 리스트
		UNLOCK_MUT_(mutrt);
		sr->srReset();

		return sr;
	}
	template<typename CT>
	void srPut(SignalR *sr)
	{
		Tracker *trk;

		LOCK_MUT_(mutrt);
		//LOCK_MUT_(sr->srMut);
		trkPut<CT>(sr->lreqTrk, false);
		//UNLOCK_MUT_(sr->srMut);
		APPEND_LIST(srPool, sr);
		UNLOCK_MUT_(mutrt);
	}
	void onSingle(bool on)
	{
		singleThr = on;
	}
	void rtRequest(Tracker *trk, intt it = 0, bool first_class = false)
	{
		TrackTh *tth = tracks[it];

		LOCK_MUT_(mutrt);
		if(first_class) {
			HEAD_LIST(tth->trkReq, trk);
		} else APPEND_LIST(tth->trkReq, trk);
		if(tth->wntrt) {
			if(singleThr == false || tth->rntrt == 0) SIG_LSEM_(tth->semrt);//요청 대기하고 있는 쓰레드가 있으면 시그널
		}
		UNLOCK_MUT_(mutrt);
	}
	void rtDispatcher(intt it = 0)
	{
		TrackTh *tth = tracks[it];
		Tracker *trk;
		intt cur_gid = -1;
		TCxtArray *tcxt_arr = nullptr;

		if(tcxtThread) {
			tcxt_arr = (TCxtArray *)malloc(sizeof(TCxtArray));
			tcxt_arr->initTCxtArray(numGpu);
		}
		while(1) {
			try {
				LOCK_MUT_(mutrt);
				while(1) {
					if(tth->trkReq) {
						GET_LIST(tth->trkReq, trk);
						tth->rntrt++;
						if(trk->execGpu) {//rdtei. 매트릭스 연산 병렬 수행용 RootTrack 실행일때 수행파트, 
							if(trk->gidTrc != cur_gid) {//graph수행용 ThreadTrack에서 설정된  
								//printf("mat parall trk_type %d %d %d\n", trk->trkType, cur_gid, trk->gidTrc);
								cur_gid = trk->gidTrc;//그라운드 아이디로 포커스됨
								CudaDevSet(cur_gid);//매트릭스 연산 수행 쓰레드와 장비 공유된다.
							}
						}
						UNLOCK_MUT_(mutrt);
						if(trk->rtShutdown) return;//전채 종료
						trk->tracking(tcxt_arr);
						LOCK_MUT_(mutrt);
						tth->rntrt--;
					} else {
						tth->wntrt++;
						UNLOCK_MUT_(mutrt);
						WAIT_LSEM_(tth->semrt);
						LOCK_MUT_(mutrt);
						tth->wntrt--;
					}
				}
			} catch(FaultObj eo) {
				printf("dispatcher error\n%s", eo.fltmsg);
				LOCK_MUT_(mutrt);
				tth->rntrt--;
				UNLOCK_MUT_(mutrt);
			}
		}
		if(tcxtThread) {
			tcxt_arr->rmTCxtArray();
			free(tcxt_arr);
		}
	}
	void rtBoot(intt cnt_rt, intt tid = 0)
	{
		struct RThreadArg *ra = (struct RThreadArg *)rutralloc(sizeof(struct RThreadArg));
		ra->ta_rt = this;
		ra->track_id = tid;

		nDistri = cnt_rt;
		for(intx i = 0;i < cnt_rt; i++) {
			xthread_create((void *)ThrRutra, ra);
		}
		xthread_create((void *)dbg_ThrRutra, this);
	}
};
#define NANO_CHECK 1
//#define SEC_CHECK 1
#ifdef NANO_CHECK
typedef chrono::system_clock::time_point chron_t;
#define chron_begin(lap, mtx) \
	chron_t lap;\
	if(mtx->lapType == 1) lap = chrono::system_clock::now()

#define _chron_end(lap) ((chrono::nanoseconds)(chrono::system_clock::now() - lap)).count()
#define chron_end(lap, imtx, rmtx, msg, msg2, n) \
	if(imtx->lapType == 1) printf("%s[%d][%d] %s: %lld\n", msg, MTX_SIZE(imtx), n, msg2, _chron_end(lap))

#define chron_begin2(lap, t, mtx) \
	chron_t lap;\
	longt t;\
	if(mtx->lapType == 1) lap = chrono::system_clock::now()
#define _chron_end2(lap, t) (t = ((chrono::nanoseconds)(chrono::system_clock::now() - lap)).count())
#define chron_end2(lap, pmtx, smtx, rmtx, msg, msg2, n, t, tp, bw, aop) \
	if(pmtx->lapType == 1) printf("%s[%d][%d][%d] type: %d bw: %d op: %d %s: %lld\n", msg, MTX_SIZE(pmtx), (smtx ? MTX_SIZE(smtx) : 1), n, tp, bw, aop, msg2, _chron_end2(lap, t))
#elif SEC_CHECK
#define chron_begin(lap, mtx) \
	unit lap;\
	if(mtx->lapType == 1) lap = xucurrenttime()

#define _chron_end(lap) (xucurrenttime() - lap) / 1000000.0
#define chron_end(lap, imtx, rmtx, msg, msg2, n) \
	if(imtx->lapType == 1) printf("%s[%d][%d] %s: %f\n", msg, MTX_SIZE(imtx), n, msg2, _chron_end(lap))

#define chron_begin2(lap, t, mtx) \
	unit lap;\
	longt t;\
	if(mtx->lapType == 1) lap = xucurrenttime()
#define _chron_end2(lap, t) (t = (xucurrenttime() - lap) / 1000000.0)
#define chron_end2(lap, pmtx, smtx, rmtx, msg, msg2, n, t, tp, bw, aop) \
	if(pmtx->lapType == 1) printf("%s[%d][%d][%d] type: %d bw: %d op: %d %s: %f\n", msg, MTX_SIZE(pmtx), (smtx ? MTX_SIZE(smtx) : 1), n, tp, bw, aop, msg2, _chron_end2(lap, t))
#else
#define chron_begin(lap, mtx)
#define chron_end(lap, imtx, rmtx, msg, msg2, n)
#define chron_begin2(lap, t, mtx)
#define chron_end2(lap, pmtx, smtx, rmtx, msg, msg2, n, t, tp, bw, aop)
#endif

#define CHECK_TIME_TEMP_VER(when) {\
    xgettime(when);\
	if(	(((when.tm_year + BASE_YEAR) == 2021 && (when.tm_mon + 1) >= 11) && \
		((when.tm_year + BASE_YEAR) == 2021 && (when.tm_mon + 1) <= 12)) || \
		(((when.tm_year + BASE_YEAR) == 2022 && (when.tm_mon + 1) >= 1) && \
		((when.tm_year + BASE_YEAR) == 2022 && (when.tm_mon + 1) <= 12))	);\
	else {\
		printf("Temporay Version expired periods! this version can use until 2021/12\n");\
	}\
}

class Cydrome;
extern Cydrome *loadCydrome(intx sem_base, intx nallocable);
extern void unloadCydrome(Cydrome *cyd);

#define CUBLAS_TRACTH	1
namespace memput {
	namespace glb {
		extern intt BLOCK_SIZE, SMALL_BLOCK, CPU_CORE_N;
		class rsc {
		public:
			static RunTrack *srutra, *brutra, *prutra;
			static intt seqTraceId, ngpudev;
			static MPClient *mpClient;
			static hmutex mutgpu;
			static Cydrome *cydrome;
			static void printDevProp(cudaDeviceProp devProp)
			{
				printf("Major revision number:         %d\n", devProp.major);
				printf("Minor revision number:         %d\n", devProp.minor);
				printf("Name:                          %s\n", devProp.name);
				printf("Total global memory:           %u\n", devProp.totalGlobalMem);
				printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
				printf("Total registers per block:     %d\n", devProp.regsPerBlock);
				printf("Warp size:                     %d\n", devProp.warpSize);
				printf("Maximum memory pitch:          %u\n", devProp.memPitch);
				printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
				for(int i = 0; i < 3; ++i)
					printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
				for(int i = 0; i < 3; ++i)
					printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
				printf("Clock rate:                    %d\n", devProp.clockRate);
				printf("Total constant memory:         %u\n", devProp.totalConstMem);
				printf("Texture alignment:             %u\n", devProp.textureAlignment);
				printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
				printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
				printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
				return;
			}
			static void BoostMemput(intt gid = 0, sytet boot_if = 0)
			{
				xtm when;

				//CHECK_TIME_TEMP_VER(when);

				if(boot_if == 0 || boot_if == 2) {
					cudaGetDeviceCount(&ngpudev);
					srutra = new RunTrack(SRUTRA_SBASE, SEM_RANAGE, ngpudev);
					brutra = new RunTrack(BRUTRA_SBASE, SEM_RANAGE, ngpudev);
					prutra = new RunTrack(PRUTRA_SBASE, SEM_RANAGE, ngpudev);
					brutra->tcxtThread = true;
					prutra->tcxtThread = true;
					srutra->rtBoot(80);//매트릭스 연산 병렬 실행.
					srutra->addTracks();//gpu cublas 실행용 trackth create
					srutra->rtBoot(2, CUBLAS_TRACTH);//2, CUBLAS_TRACTH);//gpu cublas 실행용 trackth thread 두개 할당, trackth id 는 1
					brutra->rtBoot(100);//graph 병렬 실행
					prutra->rtBoot(8);//slave track

					mpClient = new MPClient;
					mpClient->bootActor(5, 1000000);
					intt rv;
					CRE_MUT_(mutgpu, rv);
					if(rv < 0) {
						throwFault(-1, (bytex *)"cre mut fail");
					}
					//1.5 - trainable이 아니어도 저장할수있는 변수 persistant추가하기위해 oiro.관련 수정
					//		- tcfnr 수정.
					printf("Boost Memput version 2.3 gpu dev: %d start up\n", gid);
					//1.3 version에서 reduceDerive반복할수있게 하고 kerneling에서 5 case추가한 버전
					BLOCK_SIZE = 0;
					for(intt i = 0; i < ngpudev; i++)
					{
						// Get device properties
						cudaDeviceProp dev_prop;
						cudaGetDeviceProperties(&dev_prop, i);
						BLOCK_SIZE = dev_prop.maxThreadsDim[0];
						printDevProp(dev_prop);
					}
					SMALL_BLOCK = BLOCK_SIZE / 2;
					CPU_CORE_N = 4;
					cudaSetDevice(gid);
				}
				if(boot_if == 1 || boot_if == 2) {
					cydrome = loadCydrome(RT_BASE + CYDROME_SBASE, SEM_RANAGE);
				}
				//init_dbg_log();
			}
			static void *mAllocator(void)
			{
				QueryContext *qc = (QueryContext *)malloc(sizeof(QueryContext));
				memset(qc, 0x00, sizeof(QueryContext));
				InitSelPage(qc);
				return qc;
			}
			static void *ralloc(void *_qc, intt rsz)
			{
				QueryContext *qc = (QueryContext *)_qc;
				bytex *rp;

				SelAlloc(qc, rsz, rp);

				return rp;
			}
			static void *ralloc2(void *_qc, intt rsz)
			{
				QueryContext *qc = (QueryContext *)_qc;
				bytex *rp;

				SelAlloc__(bytex, qc, rsz, rp);

				return rp;
			}
			static void rrewind(void *_qc)
			{
				QueryContext *qc = (QueryContext *)_qc;
				ResetSelPage(qc);
			}
			static void rAllocator(void *_qc)
			{
				QueryContext *qc = (QueryContext *)_qc;
				ReleaseSelPage(qc);
				free(qc);
			}
			static void *flux_gener(void *tcr, intx ndim, intx dims[], intx dt)
			{
				return flux((Tracer *)tcr, ndim, dims, dt, memput::mp::apply, nullx, nullx);
			}
			static void *flux_data(void *arobj)
			{
				return ((Flux *)arobj)->begin_wp();
			}
			static Flux *rsc_combination(Flux *in, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, sytex src_make, bool one_batch)
			{//flux_gener 함수 포인터로 플럭스를 생성하여 컴비네이션 결과를 적재 한다.
				void *src_array;
				Flux *cfx = (Flux *)transform(in->begin_p(), in->end_p(), in->qType, in->fdim, in->fshape, width, stride,
					exc_contig_r, zero_pading, src_make, in->fxTcr, flux_gener, flux_data, src_array, one_batch);
				if(src_make < 0) in->fxTcr->tcr_reserve = (intx)src_array;
				return cfx;
			}
			static Flux *rsc_combination2(Flux *in, Flux *out, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, sytex src_make, bool one_batch)
			{//tcr을 널로주고 플럭스 생성 함수 포인터 자리에 기존에 생성된 out 플럭스 포인터를 설정하여 여기에 컴비네이션 결과를 적재하게 한다.
				void *src_array;
				Flux *cfx = (Flux *)transform(in->begin_p(), in->end_p(), in->qType, in->fdim, in->fshape, width, stride,
					exc_contig_r, zero_pading, src_make, nullx, (array_generfp)out, flux_data, src_array, one_batch);
				if(src_make < 0) in->fxTcr->tcr_reserve = (intx)src_array;
				return cfx;
			}
			static unit chron_get(chrono::system_clock::time_point beg)
			{
				chrono::nanoseconds nano = beg - chrono::system_clock::now();
				return nano.count();
			}
		};
	}
}
using namespace memput::glb;


