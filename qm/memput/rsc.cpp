
#include "rsc.h" //�̰��� �����ؾ� �Ѵ�.

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
#define GPU_MAX_SZ	10000000000 //100000000 GPU device�� �������� ž���ϸ� �� ������� �ʰ��ϸ� ���� �����ϰ�
//����� GPU device�� 1���̹Ƿ� ���ҵ��� �ʰ� ū�� ����.
#define GPU_PART_SZ	100000000
#define CPU_MAX_SZ	65536 //131072;//10;//BLOCK_SIZE;//
#define CPU_MIN_SZ	8192//16384
#define CPU_PART_SZ	CPU_MIN_SZ //32768
//#define CPU_CORE_N	4
#define GPU_CORE_N	1
#define GM_ALPHA	0.6 //0.5�̻� 1���� �������� �̰��� Ŭ���� gpuMultiple���� ���� ���� �۾�����.
						//���� �̰��� Ŭ���� gpuMultiple�� ���ݸ� �����Ǽ� div�� gpu�������� �ƴҶ���
						//gpu�������϶��� gpu ó�� �䱸���� ���Ͽ� ���� ���� gpu ó�� �䱸�� �ϰԵȴ�.
	//gpu�������϶��� gpu ó�� �䱸���� ���Ͽ� gpu�������� �ƴҶ� ���� ���� gpu ó�� �䱸�� �ҷ��� �̰��� �۰��Ѵ�.
	//�� 0.5�̻� 1���� �������� gpu ó������ �������� �̰��� ũ���ϰ� gpu ó������ ���Ϸ��� �̰��� �۰� �Ѵ�.
//gpuó������ �ø����� gpuMultiple�� ���� ���� gpu�� �� ����ǹǷ� gpuMultiple�� �۰��ϰ� 0.5�̻� 1���� �������� 
//GM_ALPHA�� ũ���Ѵ�. gpuMultiple�� �Ҽ����� �����̴�.
//feat_reduce = depricate
bool RunTrack::policyTrack(intt rsz, bool feat_reduce, intt feat_sz, intt div, intt &width, intt &n, sytet cpm_mode)
{
	bool exe_gpu;
	intt window, by;

	LOCK_MUT_(mutrt);
	//printf("rsz: %d GPU: %d CPU: %d ", rsz, nreqGpu, nreqCpu);
	if(cpm_mode > 0) {
		if(rsz < CPU_MIN_SZ) window = rsz;//gpu �޸� ���� ������带 �߻��ϴ� �ּ� �뷮�� cpu�� ó��
		else if(rsz > CPU_PART_SZ) {//cpu�����ε� ���� ó�� �뷮�� �Ѵ� ���� ���� ó��
			by = (rsz / CPU_PART_SZ < CPU_CORE_N ? rsz / CPU_PART_SZ : CPU_CORE_N);
			window = rsz / by;
		} else window = rsz;//cpu ���� ó�� �뷮 ���̸� �ѹ�ü cpuó��
		exe_gpu = false;
	} else if(cpm_mode < 0) {
		/*if(cpm_mode < -1 && rsz < CPU_MIN_SZ) {
			window = rsz;//gpu �޸� ���� ������带 �߻��ϴ� �ּ� �뷮�� cpu�� ó��
		} else*/ 
		if(rsz > GPU_MAX_SZ) {
			if(div > 1) {//shared multi block version dot operation, div�� share unit
				by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
				window = rsz / by;
				window = window / div ? (window / div) * div : div;//share unit ��������� ����.
			} else window = GPU_PART_SZ;
		} else window = rsz;
		exe_gpu = true;
	} else {
		if(rsz < CPU_MIN_SZ) {//gpu �޸� ���� ������带 �߻��ϴ� �ּ� �뷮�� cpu�� ó��
			window = rsz;
			exe_gpu = false;
		} else if((div > 0 && nreqGpu * gpuMultiple <= nreqCpu) || (div != PDC2 && nreqGpu < nreqCpu)) {//gpu
			if(rsz > GPU_MAX_SZ) {
				if(div > 1) {//shared multi block version dot operation, div�� share unit
					by = (rsz / GPU_PART_SZ < GPU_CORE_N ? rsz / GPU_PART_SZ : GPU_CORE_N);
					window = rsz / by;
					window = window / div ? (window / div) * div : div;//share unit ��������� ����.
				} else window = GPU_PART_SZ;
			} else window = rsz;
			exe_gpu = true;
		} else if(rsz > CPU_PART_SZ) {//cpu�����ε� ���� ó�� �뷮�� �Ѵ� ���� ���� ó��
			by = (rsz / CPU_PART_SZ < CPU_CORE_N ? rsz / CPU_PART_SZ : CPU_CORE_N);
			window = rsz / by;
			exe_gpu = false;
		} else {//cpu ���� ó�� �뷮 ���̸� �ѹ�ü cpuó��
			window = rsz;
			exe_gpu = false;
		}
	}
	//window = 5;//dot gpu ������ share unit������ ����� ���ҵǾ߱� ������ ���� ������� �����Ҽ�����.(����� Ʋ���� ������ �ǹǷ� �ٸ����� �׽�Ʈ�ϱ����� ���ҽ����ص��ȴ�.)
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
bool RunTrack::policyTrack2(intt rsz, intt &width, intt &n, sytet cpm_mode) //cpu�̸� ����ó�� ���ϰ�
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
		} else {//gpu�� sum�̶� ������ atomic add�� �ϱ⶧���� ���� ó���ص� �ȴ�.
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
//cpu�̴� gpu�̴� ����ó�� ���ϰ�, mean�� ��� gpu�� atomic add�� �ϳ� �� ������ ���� �����尡 
//����� ���ؾ� �ϴµ� ��� �����尡 �������� gpu�� �����Ͽ� ������� �˼������Ƿ� �������� �ʰ� �Ѵ�.
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
		} else {//gpu�� sum�̶� ������ atomic add�� �ϱ⶧���� ���� ó���ص� �ȴ�.
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
