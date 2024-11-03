
#include "rsc.h" //이것이 후행해야 한다.

void *Begining::operator new(size_t size, RunTrack *rt)
{
	return rt->rutralloc(size);
}
void Tracker::onSingle(bool on)
{
	rutra->onSingle(on);
}
void Tracker::ontrack(SignalR *sr, intt it, bool first_class)
{
	srGate = sr;
	LOCK_MUT_(sr->srMut);
	sr->cntReq++;
	APPEND_LIST2(sr->lreqTrk, this);
	UNLOCK_MUT_(sr->srMut);
	rutra->rtRequest(this, it, first_class);
}
#define GPU_MAX_SZ	10000000000 //100000000 GPU device를 다중으로 탑재하면 이 사이즈롤 초과하면 분할 실행하게
//현재는 GPU device가 1개이므로 분할되지 않게 큰수 설정.
#define GPU_PART_SZ	100000000
#define CPU_MAX_SZ	65536 //131072;//10;//BLOCK_SIZE;//
#define CPU_MIN_SZ	8192//16384
#define CPU_PART_SZ	CPU_MIN_SZ //32768
//#define CPU_CORE_N	4
#define GPU_CORE_N	1
#define GM_ALPHA	0.6 //0.5이상 1이하 범위에서 이값이 클수록 gpuMultiple값의 증가 폭은 작아진다.
						//따라서 이값이 클수록 gpuMultiple가 조금만 증가되서 div가 gpu전용잡이 아닐때도
						//gpu전용잡일때의 gpu 처리 요구율과 비교하여 조금 적게 gpu 처리 요구를 하게된다.
	//gpu전용잡일때의 gpu 처리 요구율과 비교하여 gpu전용잡이 아닐때 많이 적게 gpu 처리 요구를 할려면 이값을 작게한다.
	//즉 0.5이상 1이하 범위에서 gpu 처리량을 느릴려면 이값을 크게하고 gpu 처리량을 줄일려면 이값을 작게 한다.
//gpu처리량을 늘릴려면 gpuMultiple이 작을 수록 gpu로 더 실행되므로 gpuMultiple을 작게하고 0.5이상 1이하 범위에서 
//GM_ALPHA을 크게한다. gpuMultiple은 소수이하 범위이다.
//feat_reduce = depricate
bool RunTrack::policyTrack(intt rsz, bool feat_reduce, intt feat_sz, intt div, intt &width, intt &n, sytet cpm_mode)
{
	bool exe_gpu;
	intt window, by;

	LOCK_MUT_(mutrt);
	//printf("rsz: %d GPU: %d CPU: %d ", rsz, nreqGpu, nreqCpu);
	if(cpm_mode > 0) {
		if(rsz < CPU_MIN_SZ) window = rsz;//gpu 메모리 복사 오버헤드를 발생하는 최소 용량은 cpu로 처리
		else if(rsz > CPU_PART_SZ) {//cpu실행인데 분할 처리 용량을 넘는 것은 분할 처리
			by = (rsz / CPU_PART_SZ < CPU_CORE_N ? rsz / CPU_PART_SZ : CPU_CORE_N);
			window = rsz / by;
		} else window = rsz;//cpu 분할 처리 용량 내이면 한번체 cpu처리
		exe_gpu = false;
	} else if(cpm_mode < 0) {
		/*if(cpm_mode < -1 && rsz < CPU_MIN_SZ) {
			window = rsz;//gpu 메모리 복사 오버헤드를 발생하는 최소 용량은 cpu로 처리
		} else*/ 
		if(rsz > GPU_MAX_SZ) {
			if(div > 1) {//shared multi block version dot operation, div는 share unit
				by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
				window = rsz / by;
				window = window / div ? (window / div) * div : div;//share unit 배수단위로 나눔.
			} else window = GPU_PART_SZ;
		} else window = rsz;
		exe_gpu = true;
	} else {
		if(rsz < CPU_MIN_SZ) {//gpu 메모리 복사 오버헤드를 발생하는 최소 용량은 cpu로 처리
			window = rsz;
			exe_gpu = false;
		} else if((div > 0 && nreqGpu * gpuMultiple <= nreqCpu) || (div != PDC2 && nreqGpu < nreqCpu)) {//gpu
			if(rsz > GPU_MAX_SZ) {
				if(div > 1) {//shared multi block version dot operation, div는 share unit
					by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
					window = rsz / by;
					window = window / div ? (window / div) * div : div;//share unit 배수단위로 나눔.
				} else window = GPU_PART_SZ;
			} else window = rsz;
			exe_gpu = true;
		} else if(rsz > CPU_PART_SZ) {//cpu실행인데 분할 처리 용량을 넘는 것은 분할 처리
			by = (rsz / CPU_PART_SZ < CPU_CORE_N ? rsz / CPU_PART_SZ : CPU_CORE_N);
			window = rsz / by;
			exe_gpu = false;
		} else {//cpu 분할 처리 용량 내이면 한번체 cpu처리
			window = rsz;
			exe_gpu = false;
		}
	}
	//window = 5;//dot gpu 실행은 share unit사이즈 배수로 분할되야기 때문에 임의 사이즈로 분할할수없다.(결과만 틀리고 실행은 되므로 다른것을 테스트하기위해 분할실행해도된다.)
	if(exe_gpu) {
		//printf(" G\n");
		nreqGpu += rsz;
	} else {
		//printf(" C\n");
		nreqCpu += rsz;
	}
	if(rsz == window) {
		n = 1;
		width = rsz;
	} else {
		if(feat_sz) {
			width = (window / feat_sz) * feat_sz;
			if(width == 0) width = feat_sz;
		} else width = window;
		n = (rsz % width ? (rsz / width) + 1 : rsz / width);
	}
	UNLOCK_MUT_(mutrt);

	return exe_gpu;
}
bool RunTrack::policyTrack2(intt rsz, intt &width, intt &n, sytet cpm_mode) //cpu이면 분할처리 안하게
{
	bool exe_gpu;

	LOCK_MUT_(mutrt);
	if(cpm_mode > 0) {
		width = rsz;
		exe_gpu = false;
	} else if(cpm_mode < 0) {
		if(rsz < GPU_PART_SZ) width = rsz;
		else {
			intt by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
			width = rsz / by;
		}
		exe_gpu = true;
	} else {
		if(rsz < GPU_PART_SZ) {
			width = rsz;
			exe_gpu = false;
		} else {//gpu는 sum이라도 연산을 atomic add로 하기때문에 분할 처리해도 된다.
			intt by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
			width = rsz / by;
			exe_gpu = true;
		}
	}
	if(exe_gpu) nreqGpu += rsz;
	else nreqCpu += rsz;

	n = (rsz % width ? (rsz / width) + 1 : rsz / width);
	UNLOCK_MUT_(mutrt);

	return exe_gpu;
}
//cpu이던 gpu이던 분할처리 안하게, mean의 경우 gpu가 atomic add로 하나 맨 마지막 분할 쓰레드가 
//평균을 구해야 하는데 어느 쓰레드가 마지막에 gpu를 점유하여 실행될지 알수없으므로 분할하지 않게 한다.
bool RunTrack::policyTrack3(intt rsz, intt &width, intt &n, bool mean, sytet cpm_mode) 
{
	bool exe_gpu;

	LOCK_MUT_(mutrt);
	if(cpm_mode > 0) {
		width = rsz;
		exe_gpu = false;
	} else if(cpm_mode < 0) {
		if(rsz < GPU_PART_SZ || mean) width = rsz;
		else {
			intt by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
			width = rsz / by;
		}
		exe_gpu = true;
	} else {
		if(rsz < GPU_PART_SZ) {
			width = rsz;
			exe_gpu = false;
		} else {//gpu는 sum이라도 연산을 atomic add로 하기때문에 분할 처리해도 된다.
			intt by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
			width = rsz / by;
			exe_gpu = true;
		}
	}
	if(exe_gpu) nreqGpu += rsz;
	else nreqCpu += rsz;

	n = (rsz % width ? (rsz / width) + 1 : rsz / width);
	UNLOCK_MUT_(mutrt);

	return exe_gpu;
}
thrret ThrRutra(thrarg arg)
{
	struct RThreadArg *ra = (struct RThreadArg *)arg;
	ra->ta_rt->rtDispatcher(ra->track_id);

	return 0;
}
thrret dbg_ThrRutra(thrarg arg)
{
	struct RThreadArg *ra = (struct RThreadArg *)arg;

	while(1) {
		xsleep(1);
	}
	return 0;
}

namespace memput {
	namespace glb {
		intt BLOCK_SIZE, SMALL_BLOCK, CPU_CORE_N;
		RunTrack *rsc::srutra = nullptr;
		RunTrack *rsc::brutra = nullptr;
		RunTrack *rsc::prutra = nullptr;
		intt rsc::seqTraceId = 0;
		intt rsc::ngpudev = 0;
		MPClient *rsc::mpClient = nullptr;
		hmutex rsc::mutgpu = 0;
		Cydrome *rsc::cydrome = nullptr;
	}
}
intt dbg_cnt = 0;
intt dbg_counting(void)
{
	LOCK_MUT_(rsc::mutgpu);
	dbg_cnt++;
	UNLOCK_MUT_(rsc::mutgpu);
	printf("dbg count: %d\n", dbg_cnt);
	return dbg_cnt;
}
