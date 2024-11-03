#pragma once

#include "rsc.h"

#define NONET_TP		0	//타입이 없음
#define BYTET_TP		1
#define SHORTT_TP		2
#define FLOATT_TP		3
#define INTT_TP			4
#define LONGT_TP		5
#define DOUBLET_TP		6

typedef struct VectorTag_ {
	vector<Flux *> *pvector;
	struct VectorTag_ *ptrPrev;
} VectorTag;

typedef struct _ShadowCap {
	intt didshadow, shadowsz;
	void *devShadow;
	struct _ShadowCap *ptrLeft, *ptrRight;
} ShadowCap;

class Capsule : public Typer {
public:
	sytet Tgener = GEN_T_CAP;
	intt dbgIdShadow;
	Flux *vcaps;
	ShadowCap *dataShadow, *gradShadow;//didshadow, shadowsz는 동일
	Capsule *ptrLeft, *ptrRight;
	void allocShadow(intt sz, intt gid);
	void freeShadow(void);
	void setShadow(intt gid, bool grad = false);
};
class Apply;
class Matrixr;
typedef struct ApCap_ {
	Apply *vapply;
	struct ApCap_ *ptrLeft, *ptrRight;
} ApCap;
class Baper;
#define NAME_LENG	2048
#define TRACER(trc) ((Trace *)trc)
class Trace : public Tracer {
public:
	void *hmalc;
	intt colorVersion, bwVersion;
	sytet prompt;//즉시실행, 매번 구조체 생성, Trace를 delete 단계 api를 제공하여 일반 프로그램과 같이 변수 생명주기 관리해야 함.
	sytet lapType;
	sytet cpmMode;
	bool loadedWeight, fastshape, devArrange;
	Apply *trainOpt;
	NameScope *nsTop, *nsFocus, *nsOrder;
	VectorTag *vtLast;
	Matrixr *mtxlist;
	ApCap *lapply, *poolApc;
	Capsule *lorigin, *poolCap;
	hmutex mutTrc, mutMerge, mutArrange;
	bytet *trcname;
	intt idTrace, seqNscope;
	Flux *lstTarget;
	Baper *hbaper;
	intt npAbrib, tcrArrange;
	intt didTrace, sidTrace;//트래이스가 초기화 되는 쓰레드에서 설정한 메인 장비 포커스
	TCxtArray *tcxtarr;//prompt일때 그래프 생성하며 바로 실행할때 사용.
	intt nslavTrc, nexeSlav, batchPart, ibatchSize;
	//intt nfwFork, nbwFork, ifwFork, ibwFork;
	Trace *mastTrc;
	SignalR *mastsr;
	Trace *lstSlav, *ptrLeft, *ptrRight;//lstSlav연결 용
	Capsule *trainWeight;
	bytet rPath[NAME_LENG];
	longt DEV_AVAIL_MARGIN;
	//anet 관련 member
	floatt contigusRate, learningRate;
	intt bandwidth, spotlat;
	intt kernelSize, strideJump;
	intt convolving;//커널 사이즈가 8이면 half(2), quard(4), octa(8) 세가지가 허용되고(octa면 1/8이므로 더 축소될수없다)
					//커널 사이즈가 16이면 16으로 나누는 4가지가 허용된다. 커널 사이즈는 완전 조합을 생성하므로 무한정 늘일수가 없다.
	intt sourceInterv, compbot;
	intt nlearnStop;
	floatt elearnStop, lowbound, relualpha, rkernal2d;
	sytet gradby, gpugear, derivefeat;
	sytet dotType, zeroPadding, elasticHidden, ebatch_t, dot_tDec, actfType, actfAtt;
	sytet spotOpt, hidPrint, bpPrint, attsign, optType, completeConvolve, stratusOpt, stratusOpt2;
	sytet att_sDec, layer_norm, zstep, poskern, rderivet, dualloss, nblockChain, resiopt;
	sytet positional, dualChain, ehook, frozenwgt, eternity, enckern, samplingInterv, dualEmbed;
	bool dualStep, outback, tcrCublas, softfill, directMax;
	bool printBuilding, printLoad, endecode, trcpart, intergate, dualatten, resiFirst;
	bool directExec, pathPrint, ortho_single, reducea, inplus, softmdot, dual_lnorm, stracode, predlog;
	intt nbyout, slatent, sembedn, rseed, multihead, udreduce, icodefz, nbatch2pseq;
	Flux *bygate, *gotok, *endtok;//두개 토큰은 추론 과정에서만 사용
	intt numdid, *didIndice;
	//이하 dynagen변수
	sytet overlapSlide, slTrainInterv, slPredInterv, concurtrain, szDualReduce, szkernDualNest;
	bool scoopinner, batchloss, printloss, autoRegress, outerMask, externFinal;
	intt dgenmgc, inzbound, tarzbound, szslide2d, szslide1d, dualdff;
	floatt rTrainBound;
	bytet trcmsg[16];//빌드 타임 디버깅 목적
	Trace(sytet stepw, const bytet *name)
	{
		bytet buf[NAME_LENG];
		
		trcmsg[0] = '\0';
		hmalc = rsc::mAllocator();
		idTrace = 0;
		cpmMode = -1;
		loadedWeight = 0;
		if(name) sprintf(buf, "%s", name);
		else sprintf(buf, "trc%d", idTrace);
		trcname = (bytet *)xalloc(strleng(buf) + 1);
		strcpy(trcname, buf);
		tcxtarr = (TCxtArray *)xalloc(sizeof(TCxtArray));
		tcxtarr->initTCxtArray(rsc::ngpudev);
		colorVersion = 0;
		idTrace = memput::glb::rsc::seqTraceId++;
		nsOrder = nsFocus = nullptr;
		seqNscope = 0;
		nsTop = namescope(nullx);
		vtLast = nullptr;
		mtxlist = nullptr;
		lapply = poolApc = nullx;
		lorigin = poolCap = nullx;
		trainWeight = nullx;
		prompt = (stepw > 0 ? 1 : 0);
		installBaper(stepw);
		intt rv; 
		CRE_MUT_(mutTrc, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre mut fail\n");
		CRE_MUT_(mutMerge, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre mut fail\n");
		CRE_MUT_(mutArrange, rv);
		if(rv < 0) throwFault(-1, (bytex *)"cre mut fail\n");
		bwVersion = 1;
		lstTarget = nullx;
		didTrace = 0;
		npAbrib = 7;
		lapType = 0;
		nslavTrc = nexeSlav = 0;
		mastTrc = nullx;
		lstSlav = nullx;
		batchPart = 10000;
		//nfwFork = nbwFork = POSITIVE_INT_MAX;
		strcpy(rPath, ".");
		ibatchSize = 0;
		gradby = 2;//텐서플로우에서 2로 나눈다.
		characterType = 0;
		characterName[0] = '\0';
		//anet 관련 member
		bandwidth = kernelSize = 8;
		contigusRate = -1;
		strideJump = -1;
		dotType = zeroPadding = elasticHidden = -1;
		convolving = 1;
		completeConvolve = 0;
		printBuilding = 0;
		ebatch_t = 2;//이 케이스가 아닌 나머지 케이스는 시퀀스사이즈가 커널사이즈의 정수배일때만 사용한다. 예를들어 시퀀스 사이즈가 56이고
					//커널사이즈가 8이면 스트라이드가 8일때 part갯수가 7이되고(56/8) convolving이 4일때 출력사이즈는
					//16이 되어 7의 배수가 되지 않으므로 다음 컨볼빙때 ebatch_t가 1 케이스로 실행되어 중간에 역전파가
					//되질 않는다. 이 케이스를 2로 실행하면 agzo)의 설명에서와 같이 zpad 3인 케이스로 실행하지 못하므로
					//이를 고려한다.
		printLoad = 0;
		dot_tDec = -1;
		endecode = 0;
		spotlat = -1;
		lowbound = 1;
		relualpha = -1;
		layer_norm = 0;
		spotOpt = -1;
		positional = 0;
		actfType = -1;
		hidPrint = -1;
		bpPrint = -2;
		directExec = 0;
		attsign = 0;
		optType = 0;
		learningRate = -1;
		pathPrint = 0;
		stratusOpt = 0;
		stratusOpt2 = 0;
		dbgStep = dbgStep2 = 0;
		sourceInterv = 1;
		trcpart = 0;
		intergate = 0;
		actfAtt = -1;
		nlearnStop = 0;
		elearnStop = 0.02;
		att_sDec = 0;
		ortho_single = 0;

		inzbound = tarzbound = 64;
		szslide2d = 8;//2d는 x*y(8*8)사이즈, 커널당 1개는 출력되야하고 최대 압축은 커널 사이즈가 8일때
		//컨볼브를 8로하면 1, 입력이 16이므로 출력은 2가되어 출력압축코드 사이즈는 정방형이 않되므로 lowbound를
		//0.5로하여 4(커널사이즈8 * 0.5)로 맞추고 4(2*2) 로서 정방형이 되게한다, 슬라이드 사이즈를 64(8*8)로
		//한다면 컨볼브 8, 로우바운드 2로 하여 출력압축코드 사이즈를 16(4*4)로 정방형이 되게 한다.
		szslide1d = 8;//1d는 x 사이즈, lowbound를 0.25로하면 압축코드사이즈는 2, 0.125는 1, 0.5는 4
		rkernal2d = 0.125;// 1/8
		overlapSlide = 1;
		slTrainInterv = 3;
		slPredInterv = 1;
		scoopinner = 0;
		batchloss = 0;
		printloss = 0;
		concurtrain = 0;
		rTrainBound = 0.007;
		nbyout = -1;
		autoRegress = 1;
		szDualReduce = 1;
		szkernDualNest = 0;
		zstep = -1;
		rderivet = 1;
		poskern = 3;
		derivefeat = 0;
		multihead = 0;
		//dualloss = 1;
		outerMask = 1;
		dgenmgc = 3;
		reducea = 0;
		rseed = -1;
		externFinal = 0;
		dualdff = 0;
		dualChain = 0;
		inplus = 0;
		nblockChain = 0;
		resiopt = 3;
		ehook = 0;
		softmdot = 0;
		udreduce = 0;
		dual_lnorm = 0;
		dualatten = resiFirst = 0;
		frozenwgt = 0;
		icodefz = 1024;
		eternity = 0;
		stracode = 0;
		predlog = 0;
		enckern = 100;//reset 의미
		nbatch2pseq = 0;
		samplingInterv = 0;
		numdid = rsc::ngpudev;//default로 전체 gpu id 설정
		didIndice = (intt *)xalloc(numdid * sizeof(intt));
		for(intt i = 0; i < numdid; i++) *(didIndice + i) = i;
		fastshape = 0;
		devArrange = 1;
		outback = 0;
		dualStep = 1;
		tcrCublas = 1;
		DEV_AVAIL_MARGIN = 134217727; //128M //24696061952//24G
		gpugear = 3;
		softfill = true;
		dualEmbed = 3;//0 - 입력만 임베딩, 1 - 출력만 임베딩, 2 - 입력 출력 따로 임메딩, 3 - 입력 출력 함께 임베딩
		compbot = -1;
		directMax = 0;
	}
	~Trace();
	void migopt(Tracer *_sor)
	{
		Trace *sor = (Trace *)_sor;

		npAbrib = sor->npAbrib;
		lapType = sor->lapType;
		cpmMode = sor->cpmMode;
		prompt = sor->prompt;
		directExec = sor->directExec;


		convolving = sor->convolving;
		printBuilding = sor->printBuilding;
		ebatch_t = sor->ebatch_t;
		strideJump = sor->strideJump;
		kernelSize = sor->kernelSize;
		contigusRate = sor->contigusRate;
		completeConvolve = sor->completeConvolve;
		printLoad = sor->printLoad;
		dotType = sor->dotType; 
		dot_tDec = sor->dot_tDec;
		endecode = sor->endecode;
		zeroPadding = sor->zeroPadding;
		spotlat = sor->spotlat; 
		lowbound = sor->lowbound;
		relualpha = sor->relualpha;
		layer_norm = sor->layer_norm;
		spotOpt = sor->spotOpt;
		positional = sor->positional;
		actfType = sor->actfType;
		hidPrint = sor->hidPrint;
		bpPrint = sor->bpPrint;
		attsign = sor->attsign;
		optType = sor->optType;
		learningRate = sor->learningRate;
		pathPrint = sor->pathPrint;
		dbgStep = sor->dbgStep;
		stratusOpt = sor->stratusOpt;
		stratusOpt2 = sor->stratusOpt2;
		sourceInterv = sor->sourceInterv;
		trcpart = sor->trcpart;
		intergate = sor->intergate;
		actfAtt = sor->actfAtt;
		nlearnStop = sor->nlearnStop;
		elearnStop = sor->elearnStop;
		att_sDec = sor->att_sDec;
		ortho_single = sor->ortho_single;
		inzbound = sor->inzbound;
		tarzbound = sor->tarzbound;
		szslide2d = sor->szslide2d;
		szslide1d = sor->szslide1d;
		overlapSlide = sor->overlapSlide;
		slTrainInterv = sor->slTrainInterv;
		slPredInterv = sor->slPredInterv;
		scoopinner = sor->scoopinner;
		batchloss = sor->batchloss;
		printloss = sor->printloss;
		concurtrain = sor->concurtrain; 
		rTrainBound = sor->rTrainBound; 
		bandwidth = sor->bandwidth; 
		szDualReduce = sor->szDualReduce; 
		szkernDualNest = sor->szkernDualNest; 
		zstep = sor->zstep;
		rderivet = sor->rderivet;
		poskern = sor->poskern; 
		derivefeat = sor->derivefeat; 
		multihead = sor->multihead; 
		//dualloss = sor->dualloss; 
		outerMask = sor->outerMask; 
		dgenmgc = sor->dgenmgc; 
		reducea = sor->reducea; 
		rseed = sor->rseed; 
		externFinal = sor->externFinal;
		dualdff = sor->dualdff;
		dualChain = sor->dualChain;
		inplus = sor->inplus; 
		nblockChain = sor->nblockChain; 
		resiopt = sor->resiopt; 
		ehook = sor->ehook; 
		softmdot = sor->softmdot;
		udreduce = sor->udreduce;
		dual_lnorm = sor->dual_lnorm; 
		dualatten = sor->dualatten;
		resiFirst = sor->resiFirst; 
		frozenwgt = sor->frozenwgt; 
		icodefz = sor->icodefz;
		eternity = sor->eternity; 
		stracode = sor->stracode; 
		predlog = sor->predlog; 
		enckern = sor->enckern;
		rkernal2d = sor->rkernal2d; 
		nbatch2pseq = sor->nbatch2pseq; 
		samplingInterv = sor->samplingInterv;
		numdid = sor->numdid;
		for(intt i = 0; i < numdid; i++) *(didIndice + i) = *(sor->didIndice + i);
		didTrace = sor->didTrace; 
		fastshape = sor->fastshape; 
		devArrange = sor->devArrange; 
		outback = sor->outback;
		dualStep = sor->dualStep; 
		gradby = sor->gradby; 
		tcrCublas = sor->tcrCublas; 
		DEV_AVAIL_MARGIN = sor->DEV_AVAIL_MARGIN;
		gpugear = sor->gpugear; 
		softfill = sor->softfill; 
		dualEmbed = sor->dualEmbed;
		compbot = sor->compbot;
		directMax = sor->directMax;
	}
	/*void incFwFork(void)
	{
		LOCK_MUT_(mutTrc);
		if(++ifwFork > nfwFork) {
			exit(1);
		}
		UNLOCK_MUT_(mutTrc);
	}
	void incBwFork(void)
	{
		LOCK_MUT_(mutTrc);
		if(++ibwFork > nbwFork) {
			exit(1);
		}
		UNLOCK_MUT_(mutTrc);
	}*/
	/*void chkFwFork(SignalR *sr)
	{
		if(nfwFork == POSITIVE_INT_MAX) {
			xsleep(1);
			nfwFork = ifwFork;
		}
	}
	void chkBwFork(SignalR *sr)
	{
		if(nbwFork == POSITIVE_INT_MAX) {
			xsleep(1);
			nbwFork = ibwFork;
		}
	}*/
	void setmsg(bytet *msg)
	{
		if(msg == nullx) trcmsg[0] = '\0';
		else strcpy(trcmsg, msg);
	}
	void directx(bool on);
	void sizeBatch(intt sz)
	{
		batchPart = sz;
	}
	Tracer *division(void);
	void infeed(Flux *feed_fx, void *pdat, sytet tsrc, intt sz);
	void *xalloc(intt rsz)
	{
		return rsc::ralloc(hmalc, rsz);
	}
	void *bxalloc(intt rsz)
	{
		LOCK_MUT_(mutTrc);
		void *p = rsc::ralloc(hmalc, rsz);
		UNLOCK_MUT_(mutTrc);

		return p;
	}
	void installBaper(sytet stepw);
	void freeBaper(void);
	void portingGraph(Tracer *target);
	Flux *getFlux(Flux *sfx);
	Capsule *getCap(void)
	{
		Capsule *cap;

		if(poolCap) {
			GET_LIST(poolCap, cap);
			return cap;
		}
		return (Capsule *)xalloc(sizeof(Capsule));
	}
	void putCap(Capsule *cap_list)
	{
		if(cap_list == nullx) return;
		CAT_LIST(Capsule, poolCap, cap_list);
	}
	ApCap *getApc(void)
	{
		ApCap *cap;

		if(poolApc) {
			GET_LIST(poolApc, cap);
			return cap;
		}
		return (ApCap *)xalloc(sizeof(ApCap));
	}
	void putApc(ApCap *cap_list)
	{
		if(cap_list == nullx) return;
		CAT_LIST(ApCap, poolApc, cap_list);
	}
	NameScope *namescope(const bytet *nsm, bool reuse = 0)
	{
		NameScope *nsc;// = nsFocus->ptrChild;
		bytet buf[NAME_LENG];
		//for(;nsc; nsc = nsc->ptrRight) {
		//	if(!strcmp(nsc->nsName, nsm)) {
		//		nsFocus = nsc;
		//		return;
		//	}
		//}
		if(nsFocus && nsFocus->reuseScope) reuse = 1;
		if(nsFocus && nsm) {
			for(nsc = nsFocus->ptrChild;nsc; nsc = nsc->ptrRight) {
				if(!strcmp(strchr(nsc->nsName, ';') + 1, nsm)) break;
			}
			if(nsc) {
				if(reuse == 0) throwFault(-1, "exist name %s\n", nsm);
				nsFocus = nsc;
				return nsc;
			}
			sprintf(buf, "%s;%s", trcname, nsm);
		} else sprintf(buf, "%s;%d", trcname, seqNscope);
		nsc = (NameScope *)xalloc(sizeof(NameScope));
		memset(nsc, 0x00, sizeof(NameScope));
		nsc->nsName = (bytet *)xalloc(strleng(buf) + 1);
		strcpy(nsc->nsName, buf);
		nsc->reuseScope = reuse;
		seqNscope++;
		nsc->nameWeights = nsc->anonymWeights = nullptr;
		nsc->ptrParent = nsFocus;
		nsc->ptrChild = nullptr;
		if(nsFocus) APPEND_LIST(nsFocus->ptrChild, nsc);
		APPEND_LIST2(nsOrder, nsc);
		nsFocus = nsc;
		return nsc;
	}
	NameScope *namescope(intt i, bool reuse = 0)
	{
		bytet i_name[3];
		sprintf(i_name, "%d", i);
		return namescope((const bytet *)i_name, reuse);
	}
	void endscope(void)
	{
		if(nsFocus->ptrParent == nullx) throwFault(-1, "top abscent\n");

		if(nsFocus->ptrParent) nsFocus = nsFocus->ptrParent;
	}
	bytet *namewrite(const bytet *fname)
	{
		bytet buf[NAME_LENG], *p;
		if(fname) sprintf(buf, "%s/%s", nsFocus->nsName, fname);
		else sprintf(buf, "%s/%d", nsFocus->nsName, nsFocus->idNameFx);
		p = (bytet *)xalloc(strleng(buf) + 1);
		strcpy(p, buf);
		nsFocus->idNameFx++;
		return p;
	}
	NameScope *findnsc_depth(NameScope *nsc, bytet *nsm)
	{
		for(;nsc; nsc = nsc->ptrRight) {
			if(nsc->ptrChild) {
				NameScope *fsc = findnsc_depth(nsc->ptrChild, nsm);
				if(fsc) return fsc;
			}
			if(!strcmp(strchr(nsc->nsName, ';') + 1, nsm)) break;
		}
		return nsc;
	}
	NameScope *findnsc(bytet *nsm, sytet root)
	{
		NameScope *nsc;

		if(root) {//트레이서의 모든 네임스코프 검색
			for(nsc = nsOrder;nsc; nsc = nsc->ptrRight2) {
				if(!strcmp(strchr(nsc->nsName, ';') + 1, nsm)) break;
			}
			return nsc;
		} else {//현 포커스 네임스코프와 그 이하 네임스코프에서 주어진 이름의 네임스코프 검색
			if(!strcmp(strchr(nsFocus->nsName, ';') + 1, nsm)) return nsFocus;
			return findnsc_depth(nsFocus->ptrChild, nsm);
		}
	}
	Flux *findfxns(bytet *fname) //현 네임스코프에서 재사용 가중치 검색
	{
		WeightList *wl = nsFocus->nameWeights;
		for(;wl; wl = wl->ptrRight) {
			char *p = strchr(wl->weightFx->fxName, '/') + 1;
			if(!strcmp(strchr(wl->weightFx->fxName, '/') + 1, fname)) break;
		}
		if(wl) {
			return wl->weightFx;
		} else return nullx;
	}
	void get_trainvar(NameScope *nsc, vector<Flux *> *rsp)
	{
		for(WeightList *wl = nsc->nameWeights; wl; wl = wl->ptrRight) {
			if(wl->weightFx->fxType == trainable) rsp->push_back(wl->weightFx);//oiro.
		}
		if(nsc->ptrChild) get_trainvar(nsc->ptrChild, rsp);
		if(nsc->ptrRight) get_trainvar(nsc->ptrRight, rsp);
	}
	vector<Flux *> *trainvar(NameScope *nsc) //주어진 네임스코프와 그 이하에서 train var(이믈을 명시안해도 trainable이면) 리스팅
	{
		vector<Flux *> *rsp = new vector<Flux *>;

		listv(rsp);

		if(nsc == nullx) nsc = nsFocus;
		for(WeightList *wl = nsc->nameWeights; wl; wl = wl->ptrRight) {
			if(wl->weightFx->fxType == trainable) rsp->push_back(wl->weightFx);//oiro.
			rsp->push_back(wl->weightFx);
		}
		if(nsc->ptrChild) get_trainvar(nsc->ptrChild, rsp);

		return rsp;
	}
	void listw(Flux *fx, bool anonym)
	{
		WeightList *wl = (WeightList *)xalloc(sizeof(WeightList));

		wl->weightFx = fx;
		if(anonym) {
			APPEND_LIST(nsFocus->anonymWeights, wl);
		} else APPEND_LIST(nsFocus->nameWeights, wl);
	}
	Capsule *elistw(bool anonym, intt fx_type)
	{
		NameScope *nsc = nsOrder;
		WeightList *wl;
		Capsule *cap, *lcap = nullx;

		for(;nsc; nsc = nsc->ptrRight2) {
			for(wl = (anonym ? nsc->anonymWeights : nsc->nameWeights);wl; wl = wl->ptrRight) {
				//if(wl->weightFx->backwv != bwVersion) {//오차함수에 입력되는 타겟 fx는 입력으로 리스팅되지 않아
				//	continue;//vsync에서 버전 설정되지 않으므로 여기 버전 체크에서 걸려 스킵된다.(정상)
				//}
				if(fx_type >= 0 && wl->weightFx->fxType != fx_type) continue;
				cap = (Capsule *)xalloc(sizeof(Capsule));//여기서 할당된는 cap은 divObj 설정될 필요없으므로 단순 할당.
				cap->vcaps = wl->weightFx;
				APPEND_LIST(lcap, cap);
			}
		}
		return lcap;
	}
	Capsule *persistw(void) //oiro.
	{
		NameScope *nsc = nsOrder;
		WeightList *wl;
		Capsule *cap, *lcap = nullx;

		for(; nsc; nsc = nsc->ptrRight2) {
			for(wl = nsc->nameWeights; wl; wl = wl->ptrRight) {
				if(wl->weightFx->fxType > trainable) continue;
				cap = (Capsule *)xalloc(sizeof(Capsule));//여기서 할당된는 cap은 divObj 설정될 필요없으므로 단순 할당.
				cap->vcaps = wl->weightFx;
				APPEND_LIST(lcap, cap);
			}
			for(wl = nsc->anonymWeights; wl; wl = wl->ptrRight) {
				if(wl->weightFx->fxType > trainable) continue;
				cap = (Capsule *)xalloc(sizeof(Capsule));//여기서 할당된는 cap은 divObj 설정될 필요없으므로 단순 할당.
				cap->vcaps = wl->weightFx;
				APPEND_LIST(lcap, cap);
			}
		}
		return lcap;
	}
	void reset_grads(bool anonym)
	{
		NameScope *nsc = nsOrder;
		WeightList *wl;
		for(;nsc; nsc = nsc->ptrRight2) {
			for(wl = (anonym ? nsc->anonymWeights : nsc->nameWeights);wl; wl = wl->ptrRight) {
				if(wl->weightFx->backwv != bwVersion) {//오차함수에 입력되는 타겟 fx와 옵티마이져의 출력fx는 ap의 입력으로
					continue;//리스팅되지 않아 vsync에서 버전 설정되지 않으므로 여기 버전 체크에서 걸려 스킵된다.(정상)
				}
				if(didTrace != wl->weightFx->groundid(1)) {//backwardDevArrange에의해 가중치는
					didTrace = wl->weightFx->groundid(1); //모두 그라운드로 복사된다.
					CudaDevSet(didTrace);//트레이스의 run함수 수행 쓰레드와 장비 공유된다.
				}
				wl->weightFx->resetGrad();
				wl->weightFx->ibRefer = 0;
				//printf("2: %p: %s\n", wl->weightFx, wl->weightFx->fxName);
			}
		}
	}
	void resetGrads(void)
	{
		reset_grads(true);
		reset_grads(false);
	}
	void listv(vector<Flux *> *pvec)
	{
		VectorTag *vtag = (VectorTag *)xalloc(sizeof(VectorTag));

		vtag->pvector = pvec;
		vtag->ptrPrev = vtLast;
		vtLast = vtag;
	}
	void listm(Matrixr *mx);
	void npset(intt n)
	{
		npAbrib = n;
	}
	void lapset(sytet lap)
	{
		lapType = lap;
	}
	void gprset(floatt d)
	{
		memput::glb::rsc::srutra->gpuMultiple = d;
	}
	void modeset(sytet cpm)
	{
		cpmMode = cpm;
	}
	Matrixr *instMatrix(Matrixr *mtx, ubytet qt, intt ndim, intt *axid, bool o_mut, intt gid = -1, Matrixr *mast = nullptr);
	void promptMode(bool on)
	{
		prompt = on;
	}
	void listOrigin(Flux *ori);
	void initArrange(void);
	void initVsync(void);
	void listWeight(void);
	void vSync(Flux *endp);
	void resetApply(void);
	void init_train(void);
	void run(Flux *target);
	void run(vector<Flux *> target);
	void run(vector<Flux *> *target);
	void reposet(bytet *rpath)
	{
		strcpy(rPath, rpath);
	}
	void setgpudev(intt gid)//특정 디비이스가 명시되지 않고 장비매모리 할당될때의 디바이스 설정
	{
		if(didTrace != gid) {
			didTrace = gid;
			CudaDevSet(gid);
		}
	}
	void focusgpudev(intt gid = -1)
	{
		if(gid < 0) gid = didTrace;

		setgpudev(gid);
	}
	void multiDevice(vector<intt> axid)
	{
		intt i = 0, gid[256];

		for(auto iter : axid) gid[i++] = iter;

		if(i > numdid) throwFault(-1, "gpu over\n");
		if(gid[0] < 0) return;//전체 디바이스 설정, default로 설정되있으므로 스킵

		numdid = i;
		for(i = 0; i < numdid; i++) *(didIndice + i) = gid[i];

		if(*(didIndice + 0) != didTrace) {//명시된 디바이스 리스트의 첫번째 아이디 설정.
			didTrace = *(didIndice + 0);//setgpudev로 설정된것은 이것으로 대채됨
			CudaDevSet(didTrace);
		}
	}
	TContext *trcCxt(intt gid)
	{
		return tcxtarr->getgputxt(gid, this);
	}
	//배치사이즈를 변경할려면 gate, go, end에 feed함수로 입력하거나 resizing5로 배치사이즈를 변경한다.
	//dual encoder실행은 chatbot예제에서 <s> 시작토큰을 데이터에서 제거하고 실행한다. 시작토큰은 0값
	//으로서 auto regression에서 시작 토큰으로 사용되기 때문에, 다이나젠에서 사용할 경우 embedim은
	//듀얼 인코더의 임베딩 사이즈로 반드시 설정하고 latent_sz, indiscret는 듀얼의 사이즐 다이나젠
	//생성자에서 명시한 것과 다르게 할 경우 설정한다. outdiscret는 다이나젠 생성자에서 듀얼 인코더의
	//것이 명시되므로 따로 설정 필요없다.
	void setbygate(Flux *gate, intt nout, Flux *go, Flux *end, intt embedim = -1, intt latent_sz = -1, intt indiscret = -1)
	{
		if(nout > gate->fshape[1]) throwFault(-1, "out size(%d) is longer than by gate(%d) error \n", nout, gate->fshape[1]);
		bygate = gate;//입력 + 추론출력 pair 혹은 추론출력, 추론 출력은 <s> + 타겟
		nbyout = nout;//추론 출력 길이, 이 값이 바이게이트 전제길이이면 여기에 입력은 없고
		//입력은 제너릭의 ingate에 주어저 제너릭의 입력 압축 망에서 압축되어 pcode로 듀얼인코더에 입력됨.
		gotok = go;//학습만 할 경우는 필요없다.
		endtok = end;//추론을 배치 단위로 할 경우는 필요없다.
		sembedn = embedim;
		slatent = latent_sz;
	}
	Flux *migbygate(Trace *src)
	{
		bygate = flux(this, src->bygate, variable);
		nbyout = src->nbyout;
		gotok = src->gotok;
		endtok = src->endtok;
		sembedn = src->sembedn;
		slatent = src->slatent;
		
		return bygate;
	}
	void traceopt(intt i, doublet v)
	{
		switch(i) {
		case 0:
			convolving = (intt)v;
			break;
		case 1:
			printBuilding = (intt)v;
			break;
		case 2:
			ebatch_t = (intt)v;
			break;
		case 3:
			strideJump = (intt)v;
			break;
		case 4:
			kernelSize = (intt)v;
			break;
		case 5:
			contigusRate = v;
			break;
		case 6:
			completeConvolve = (intt)v;//1아면 인코딩에서만 splot lat적용, 2이면 인코딩/디코딩에서 splot lat적용
			break;
		case 7:
			printLoad = (intt)v;
			break;
		case 8:
			if((intt)v == 1) BLOCK_SIZE = SMALL_BLOCK;
			else BLOCK_SIZE = 2 * SMALL_BLOCK;
			break;
		case 9:
			dotType = (intt)v;
			break;
		case 10:
			dot_tDec = (intt)v;
			break;
		case 11:
			endecode = (intt)v;
			break;
		case 12:
			spotlat = (intt)v;
			break;
		case 13:
			lowbound = v;
			break;
		case 14:
			CPU_CORE_N = (intt)v;
			break;
		case 15:
			layer_norm = (intt)v;
			break;
		case 16:
			spotOpt = (intt)v;
			break;
		case 17:
			positional = (intt)v;
			break;
		case 18:
			actfType = (intt)v;
			break;
		case 19:
			hidPrint = (intt)v;
			break;
		case 20:
			bpPrint = (intt)v;
			break;
		case 21:
			attsign = (intt)v;//spot opt가 3인 조건과 함께 2이면 모두 어텐션 수생 
			break;// 1이면 인코더-디코더 구성 네크워크(자기구조학습)에서는 실행 않고 인코더만 구성 네크워크(소스연결학습)에서만 실행.
		case 22:
			optType = (intt)v;
			break;
		case 23:
			learningRate = v;
			break;
		case 24:
			pathPrint = (intt)v;
			break;
		case 25:
			dbgStep = (sytet)v;
			break;
		case 26:
			stratusOpt = (intt)v;
			break;
		case 27:
			stratusOpt2 = (intt)v;
			break;
		case 28:
			sourceInterv = (intt)v;
			break;
		case 29:
			trcpart = (intt)v;
			break;
		case 30:
			intergate = (intt)v;
			break;
		case 31:
			actfAtt = (intt)v;
			break;
		case 32:
			nlearnStop = (intt)v;
			break;
		case 33:
			elearnStop = v;
			break;
		case 34:
			att_sDec = (intt)v;
			break;
		case 35:
			ortho_single = (intt)v;
			break;
		case 36:
			break;
		case 37:
			break;
		case 38:
			break;
		case 39:
			break;
		case 40:
			break;
		case 41:
			break;
		case 42:
			printloss = (intt)v;
			break;
		case 43:
			inzbound = (intt)v;
			break;
		case 44:
			tarzbound = (intt)v;
			break;
		case 45:
			szslide2d = (intt)v;
			break;
		case 46:
			szslide1d = (intt)v;
			break;
		case 47:
			overlapSlide = (intt)v;
			break;
		case 48:
			slTrainInterv = (intt)v;
			break;
		case 49:
			slPredInterv = (intt)v;
			break;
		case 50:
			scoopinner = (intt)v;
			break;
		case 51:
			batchloss = (intt)v;
			break;
		case 52:
			concurtrain = (intt)v;
			break;
		case 53://dynagen default 구성
			convolving = 8;//커널 사이즈가 8일때 일단 최대 컨볼빙으로서 커널당 1이 되게하고 로우바운드로 결정하게한다.
			if((intt)v == 1) lowbound = 0.125;//1d, 0.25 출력 압축코드사이즈는 2, 0.125로 설정하여 1로 할지 검토
			else if((intt)v == 2) lowbound = 0.5;//2d
			else throwFault(-1, "non def\n");
			dot_tDec = 1;//TENSOR_DOT.여기서 이 두개 옵션값을 설정할때 디코더 2차(오류) 학습(translate일때)이 
			dotType = 4;//ORTHO_DOT.//잘된다. 다른데는 기본값일때가 훨씬 학습이 잘되지만 디코더 2차학습이 느려 
			//이값을 설정한다. 따라서 인코딩만 할경우에는 기본값(둘다 0[stride dot])으로 설정하여 실행한다.
			derivefeat = 0;//듀얼인코더가 실행될때 의미, 이값을 0로 하면 feature가 히든사이즈가 
			//되어 위 설정과 일치된다. ORTHO_DOT설정이면 dynagen의 인코딩망에서 feature가 히든사이즈
			//가 되고 옵션으로 derivefeat를 1로 설정하여도 듀얼인코더에서 pcode의 feature차원 
			break; //사이즈가 derive사이즈로 자동변환되므로 후에 1로 설정해도 문제없다.
		case 54:
			rTrainBound = v;
			break; 
		case 55:
			autoRegress = v;
			break; 
		case 56:
			bandwidth = v; 
			break;
		case 57:
			szDualReduce = v;
			break;
		case 58:
			relualpha = v; 
			break;
		case 59:
			szkernDualNest = v;
			break;
		case 60:
			zstep = (intt)v;
			break;
		case 61:
			rderivet = (intt)v;
			break;
		case 62:
			poskern = (intt)v;
			break; 
		case 63:
			derivefeat = (intt)v;
			break; 
		case 64:
			multihead = (intt)v; 
			break;
		case 65:
			//dualloss = (intt)v;
			break;
		case 66:
			outerMask = (intt)v; 
			break; 
		case 67:
			dgenmgc = (intt)v;
			break;
		case 68:
			reducea = (intt)v;
			break; 
		case 69:
			rseed = (intt)v;
			break;
		case 70:
			externFinal = (intt)v;
			break;
		case 71:
			zeroPadding = (intt)v; 
			break;
		case 72:
			dualdff = (intt)v;
			break;
		case 73:
			dualChain = (intt)v;
			break;
		case 74:
			inplus = (intt)v;
			break;
		case 75:
			nblockChain = (intt)v;
			break; 
		case 76:
			resiopt = (intt)v; 
			break; 
		case 77:
			ehook = (intt)v;
			break;
		case 78:
			softmdot = (intt)v;
			break; 
		case 79:
			udreduce = (intt)v;
			break;
		case 80:
			dual_lnorm = (intt)v;
			break;
		case 81:
			dualatten = (intt)v; 
			break;
		case 82:
			resiFirst = (intt)v;
			break; 
		case 83:
			frozenwgt = (intt)v;
			break;
		case 84:
			icodefz = (intt)v; 
			break; 
		case 85:
			eternity = (intt)v;
			break;
		case 86:
			stracode = (intt)v;
			break; 
		case 87:
			predlog = (intt)v;
			break; 
		case 88:
			enckern = (intt)v;
			break;
		case 89:
			rkernal2d = v;
			break;
		case 90:
			nbatch2pseq = (intt)v;
			break;
		case 91:
			samplingInterv = (intt)v;
			break;
		case 92:
			fastshape = (intt)v;
			break;
		case 93:
			devArrange = (intt)v;
			break; 
		case 94:
			outback = (intt)v;
			break;
		case 95:
			dualStep = (intt)v; 
			break;
		case 96:
			gradby = (intt)v;
			break; 
		case 97:
			tcrCublas = (intt)v; 
			break;
		case 98:
			DEV_AVAIL_MARGIN = (longt)v; 
			break;
		case 99://0 : 초기 설정된 gpu에서만 실행, 1 : 현행 gpu에 메모리가 모자르면 다음 gpu에서 실행
			gpugear = (intt)v;//2 : split 분기에서만 다음 gpu 멀티 실행. 3 : 모든 분기마다 다음 gpu 멀티 실행
			break; 
		case 100:
			softfill = (intt)v;
			break;
		case 101:
			dualEmbed = (intt)v; 
			break;
		case 102:
			compbot = (intt)v;
			break;
		case 103:
			directMax = (intt)v;
			break;
		default:
			throwFault(-1, "non def trace option\n");
		}
	}
	void saveWeight(void);
	intt loadWeight(void);
	void truncWeight(void);
	void printWeight(void);
	Flux *tcr_combination(Flux *in, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, bool one_batch);
	Flux *tcr_combination2(Flux *in, intx width, intx stride, doublex exc_contig_r, sytet zero_pading, bool one_batch);
};
//이하 30만 스텝 테스트
/* stratus - 50% 정확도, 히든 사이즈 128, 타겟 시퀀스를 디코드1으로(32->8->32) 자기구조학습시킨후  
			타겟시퀀스 인코더 최종압축코드를 소스로 디코드2망을(8->32) 학습하고 타겟으로 소스연결망을(32->8->8) 학습, 
			디코드2망의 오차는 0.02%로 나오나 이 오차와 소스 연결망 오차가 더해서 전채 정확도는 50%정도만 나온다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);//디코더에만 적용되는 dot타압
TRACER(tcr)->traceopt(6, 1);//완전압축 옵션, 1아면 인코더 네트워크에서만 splot lat적용(완전압축), 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);//망 전체에 적용되은 dot타입
TRACER(tcr)->traceopt(26, 3);//소스인코더 연결망, 타게시퀀스 인코더-디코더 자기구조학습망에 더하여 디코드2 망을 하나 더 구성하여 학습. stratus 망 구성 옵션
TRACER(tcr)->traceopt(27, 1);//소스와 타겟의 압축코드 연결 오차함수를 0이면 소프프맥스크로스엔트로피, 1이면 mse
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 10~20% 정확도, 히든 사이즈 128, 소스연결망 인코더를 완전압축 어텐션 연결하고(32->8->8[attention]) 타겟시퀀스의 
			자기구조학습망을 인코더-디코더2로(32->8->32) 구성한다. 인코더-디코더2 자기구조학습의 오차가 줄질 않는다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(11, 1);//인코더-디코더 망에서 디코더를 디코더2로 구성
TRACER(tcr)->traceopt(6, 1);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크의 인코더에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);//인코더망이던 인코더-디코더망이던 특정옵션(어텐션, 잠재코드크기등)들을 최종 압축에서만 유효하게
TRACER(tcr)->traceopt(21, 1);//어텐션은 위 6과 함께 이값이 1아면 인코더망에서만, 2이면 인코더-디코더망의 인코더에서 적용
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 10~% 정확도, 히든 사이즈 64, 소스연결망 인코더를 완전압축 어텐션 연결하고(32->8->8[attention]) 타겟시퀀스의 
			자기구조학습망을 인코더-디코더2로(32->8->32[attention]) 구성한다. 인코더-디코더2 자기구조학습의 오차가 줄질 않는다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(11, 1);
TRACER(tcr)->traceopt(6, 1);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 1);//어텐션은 위 6에 따라 젹용 범위 결정.
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);
TRACER(tcr)->traceopt(34, 1);//디코더에서 어텐션 적용 옵션, 1이면 압축후 첫번째 디코더에서만 2이면 디코더 전체에서 어텐션 적용

// stratus - 90% 정확도, 히든 사이즈 64, 소스연결망(32->8->8) 인코더를 완전압축 연결하고 타겟시퀀스의 자기구조학습망을
			인코더-디코더로(32->8->32) 구성한다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 1);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크의 인코더에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 70% 정확도, 히든 사이즈 64, 소스연결망(32->8->8) 인코더를 완전압축 연결하고 타겟시퀀스의 자기구조학습망을
			인코더-디코더로(32->8->8->32) 구성한다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 80~90% 정확도, 히든 사이즈 64, 소스연결망(32->8->8[attention]) 인코더를 완전압축 연결하고 타겟시퀀스의 자기구조학습망을
			인코더-디코더로(32->8->8[attention]->32) 구성한다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 2);//어텐션은 위 6에 따라 젹용 범위 결정.
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.008);

// stratus - 70 ~ 80% 정확도, 히든 사이즈 64, 소스연결망 인코더를(32->8->8) 완전압축 연결하고 타겟시퀀스의 자기구조학습망을
			인코더-어텐션디코더로(32->8->32[attention]) 구성한다. 인코더-어텐션디코더 학습이 불규칙하여 0.005에 100번 연속하여
			도달하지 못해	자기구조학습망의 인코더 최종압축코드가 고정되질 않아서 소스 연결망의 정확도가 떨어진다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 1);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(26, 2);
TRACER(tcr)->traceopt(27, 1);
TRACER(tcr)->traceopt(32, -100);
TRACER(tcr)->traceopt(33, 0.005);
TRACER(tcr)->traceopt(34, 1);

//stratus - 히든 사이즈 64, 타겟시퀀스의 자기구조학습망을 인코더-디코더2로(32->8->32[attention]) 구성하고
			평가때 입력시퀀스에 복원하고자 하는 시퀀스의 최총압축코드를 넣고 예측 결과로 복원한다.
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(11, 1);
TRACER(tcr)->traceopt(6, 1);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 1);//어텐션은 위 6에 따라 젹용 범위 결정.
TRACER(tcr)->traceopt(26, -1);
TRACER(tcr)->traceopt(34, 1);



// generic - 95% 정확도, 히든 사이즈 64, (32 -> 8 -> 32)
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
//TRACER(tcr)->traceopt(6, 1);//인코더 only 네크워크가 없으므로 의미없음
TRACER(tcr)->traceopt(9, 4);

// generic - 20% 정확도, 히든 사이즈 64, 32 -> 8 -> 8 -> 32)
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);

// generic - 80% 정확도, 히든 사이즈 64, (32 -> 8 -> 8[attention] -> 32) 압축
TRACER(tcr)->traceopt(1, 1);
TRACER(tcr)->traceopt(0, 4);
TRACER(tcr)->traceopt(8, 1);//small block
TRACER(tcr)->traceopt(10, 1);
TRACER(tcr)->traceopt(6, 2);//1아면 인코더 네트워크에서만 splot lat적용, 2이면 인코더-디코더 네크워크에서 splot lat적용
TRACER(tcr)->traceopt(9, 4);
TRACER(tcr)->traceopt(16, 3);
TRACER(tcr)->traceopt(21, 2);//어텐션은 위 6에 따라 젹용 범위 결정.

*/