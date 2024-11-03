#pragma once
//injection, ejection, organail, isleloop, telepulse
#include "islet.h"
#include "misc/fio.h"

class Isleloop;
class Impulse;
class Dynagen;

class SyncInvoke : public Typer {
public:
	sytet sockid;
	Isleloop *invokeisle;
	SyncInvoke *ptrLeft, *ptrRight;
	SyncInvoke(Isleloop *isle, sytet socid)
	{
		invokeisle = isle;
		sockid = socid;
	}
};
class Socketisle : public Typer {
public:
	sytet sockid;
	Isleloop *destisle;
	Socketisle *ptrLeft, *ptrRight;
};
class Impulse : public Cell {//이미 학습된 것의 예측 결과를 다른 것의 학습 입력으로 할때 impulse로 분리하여 수행한다.
public:
	Trace *impTcr;
	QueryContext impAllocator, *qcimp;
	intt ntrainStep;//, ntrainBatch;
	SyncInvoke *synchor;
	intt sample_size, batch_size, ntrainisle, ndecisle, itrainisle, idonetrain;
	longx syncIndice;//입력이 두개 인 것들은 이 스템프가 동일해야 작동한다.
	bool tadapt, predwithloss, finalSync, runPredict;
	sytet trainstep, init_trainstep;
	Isleloop *islelist;
	Dynagen *dgenlist;
	hmutex mutimp;
	lsemx semimp;
	CydChain *cydimp;
	bytet impname[128], imppath[128];
	intt cntdgen;
	Generic accucell;// Cell *accucell;
	Tracer *accutrc;
	Impulse(Tracer *tcr, const bytet *name = nullx)
	{
		impTcr = (Trace *)tcr;//메모리 할당과 관련해서는 빌드할때만 공유되어야 한다.
		qcimp = &impAllocator;
		InitSelPage(qcimp);
		synchor = nullx;
		syncIndice = 0;
		islelist = nullx;
		dgenlist = nullx;
		tadapt = 0;
		ntrainisle = ndecisle = idonetrain = 0;
		init_trainstep = 1;
		runPredict = 0;
		intt rv;
		CRE_MUT_(mutimp, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre mut fail");
		}
		cydimp = rsc::cydrome->getcydid();
		CRE_LSEM_(semimp, 0, cydimp->sem_iden, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre sem fail\n");
		}
		predwithloss = 0;
		if(strcmp(impTcr->rPath, ".")) strcpy(imppath, impTcr->rPath);
		else strcpy(imppath, ".");
		if(name) {
			if(strlen(name) >= 128) throwFault(-1, (bytex *)"name over fail\n");
			strcpy(impname, name);
		} else strcpy(impname, "_impluse");
		cntdgen = 0;
		finalSync = 0;
		accucell.makeAnet(impTcr, 0, -1);
	}
	~Impulse();
	void *impalloc(size_t size)
	{
		bytex *rp;

		SelAlloc(qcimp, size, rp);

		return rp;
	}
	void trainStep(longx v)
	{
		ntrainStep = v;
	}
	void countDoneTrain(void)
	{
		LOCK_MUT_(mutimp);
		idonetrain++;
		UNLOCK_MUT_(mutimp);
	}
	void sigtraining(void)
	{
		if(finalSync) {//동시학습 2단계이면 마지막으로 학습 수행된 isle이 이번 스텝 학습 완료 시그널
			LOCK_MUT_(mutimp);
			if(++itrainisle == ntrainisle) {
				SIG_LSEM_(semimp);
			}
			UNLOCK_MUT_(mutimp);
		}
	}
	void waittraining(void)
	{
		if(finalSync) WAIT_LSEM_(semimp);
	}
	void regifeed(Isleloop *begin_isle, sytet socid)
	{
		SyncInvoke *s = new(impTcr) SyncInvoke(begin_isle, socid);

		APPEND_LIST(synchor, s);
	}
	void recording(void);
	void retrain(void);
	void regiisle(Isleloop *isle);
	void regidgen(Dynagen *dgen);
	void coupleisleThr(void);
	Flux *train(intt *n_train = 0);
	Flux *predict(Flux **ploss = nullx);
	Flux *loss2(void) { return nullx; }
	void accuracy(Flux *&predicts, Flux *&targets, intt discrete_out) //정확도 측정 그라프 빌드
	{//학습할때의 입력과 타겟 게이트를 사용하지 않고 따로 입출력을 두는 것은 잠재 코드 정확도를 계산할때는 
		//학습 입출력때의 데이터와 잠재코드의 형태가 틀리기 때문이다.
		accucell.accuracy(predicts, targets, discrete_out);//ipare.필요하면 아랫것으로 한다.
		//predicts = flux(accutrc, predicts, variable);
		//targets = flux(accutrc, targets, variable);
		//accucell->accuracy(predicts, targets, discrete_out);
	}
	Flux *measureAccuracy(void) //정확도 측정 그라프 실행.
	{
		return accucell.measureAccuracy();//accucell->measureAccuracy();
	}
};
#define NHNDI	0 //not head, not discrete, input
#define NHNDT	1 //not head, not discrete, target
#define NHNDC	2 //not head, not discrete, couple
#define NHOC	3 //not head, output couple
#define HID		4 //head input discrete
#define HOD		5 //head output discrete
#define HAD		6 //head input, output all discrete


#define ENCODE_ISLE	0
#define DECODE_ISLE	1
#define COUPLE_ISLE	2
class Dynagen;
class Isleloop : public Typer {
public:
	sytet isleType;
	sytet headisle, actcode, noperand, cntoper, cntwait, islebusy;
	sytet lossIndice, nprotect;
	bytex isleName[128];
	Trace *isletrc;
	Cell *islenet;//stratus: 소스 - 타겟 연결, generic: 오토인코더, decoder: 자기복원 망
	Impulse *isleimp;
	Dynagen *isledgen;
	Socketisle *isleRoute;
	longx syncurIndice, rcvsynIndice;
	intt cntTStep, nTStep, szlatent, isindisc, isoutdisc, szembede, icntTrain, icumTrain;
	floatt rLeaning, sumloss, minloss;
	intt istableMean, istableMean2;
	Flux *predloss, *islein[2], *isleout, *scoopout, *extingate, *exttargate;
	bool fintrain, sockIndice[2], tadapt, alignoper, finalisle, fowardit, neverTrained;
	Cydrome *cydisle;
	CydChain *cydcijec, *cydcpeek;
	hmutex mutisle;
	lsemx semijec, sempeek;
	Isleloop *ptrLeft, *ptrRight;

	Isleloop(Impulse *imp, Dynagen *dgen, sytet head, bytet *name);

	~Isleloop()
	{
		rmloop();
	}
	virtual void rmloop()
	{
		inject(nullx, nullx, -1, 0);//isle 종료 지시 및 쓰레드 종료때까지 대기
		CLOSE_MUT_(mutisle);
		CLOSE_LSEM_(semijec);
		cydisle->putcydid(cydcijec);
		CLOSE_LSEM_(sempeek);
		cydisle->putcydid(cydcpeek);
		delete isletrc;
	}
	//나중에 buildislet, trainislet, predislet을 독립 프로세스에 구성하게 되면 in, tar를 전송하여 주입케 한다.
	//이들 함수를 로컬에 구성하면 in, tar는 입력 플럭스(islein[2])로 설정되있으므로 다시 주입할 필요업다.
	virtual Flux *trainislet(Flux *in, Flux *tar)
	{
		return ((Generic *)islenet)->train();
	}
	virtual Flux *predislet(Flux *in, Flux **loss)
	{
		return ((Generic *)islenet)->predict(loss);
	}
	virtual void trainloop(longx sync_indice)
	{
		if(fintrain) {//현 단계 학습이 끝났으면
			predloop(sync_indice);//다음 상위 계층의 학습을 위한 평가 데이터 생산 및 상위로 포워드
			return;
		}
		Flux *r = trainislet(islein[0], islein[1]);
		finaltrain(r);
		lossout(r, "_train");
		if(isletrc->concurtrain && fowardit) {//동시학습이면 상위 연결이 있으면(디코더가 아닌)
			predloop(sync_indice);//현 평가 및 상위 포워드
		}
	}
	virtual void predloop(longx sync_indice);
	virtual Flux *restoreOrigin(Flux *dout)
	{
		return dout;
	}
	void printisle(Flux *pf, const bytet *msg)
	{
		LOCK_MUT_(isleimp->mutimp);
		printf("%s [%p] %s\n", isleName, this, msg);
		pf->printo(2, 10);
		UNLOCK_MUT_(isleimp->mutimp);
	}
	void setFinal(bool on)
	{
		if(isletrc->externFinal) neverTrained = on;
		finalisle = on;
	}
	void decreaseProtect(void);
	void finaltrain(Flux *loss);
	void lossout(Flux *loss, const bytet *action);
	void setdgencoup(Dynagen *dgen);
	void regiRouteTo(Isleloop *next, sytet socid)
	{
		Socketisle *s = new(isleimp->impTcr) Socketisle;
		s->destisle = next;
		s->sockid = socid;
		APPEND_LIST(isleRoute, s);
		if(next->isleType != DECODE_ISLE) fowardit = 1;
	}
	void eject(Flux *preds, longx sync_indice)
	{
		for(Socketisle *s = isleRoute;s; s = s->ptrRight) {
			//학습 1단계에서는 인코더에서 디코더로 eedp)에서 inject로 직접 포워드하므로 여기서는 스킵하고
			//2단계 학습 혹은 평가단계에서는 디코더로의 포워는 최종 커플 isle->최종 디코더, 디코더->디코더간에 된다.
			if(isleimp->trainstep > 0 && s->destisle->isleType == DECODE_ISLE) continue;
			//추론 과정에서 오차 획득하지 않는 옵션이면 타겟값이 주입되지 않으므로 입력만으로 실행되게 -1로 한다.
			s->destisle->inject(preds, nullx, sync_indice, isleimp->predwithloss || s->sockid == -2 ? s->sockid : -1);
		}
	}
#define CHKSIG_ISLE(snd_peek) {\
	if(snd_peek) SIG_LSEM_(sempeek);\
	if(cntwait) {\
		SIG_USEM_(semijec, cntwait);\
		cntwait = 0;\
	}\
}
	//각 isle은 각각 자신의 tracer를 가지고 오류역잔파가 독립적으로 수행되므로 입력 및 타겟 플럭스를 feedf로 복사 주입해야한다.
	//안그러면 트레이스간에 그라프가 연결된다.
	void inject(Flux *in, Flux *tar, longx sync_indice, sytet i_socket) //현 isle은 tar가 널이면 인코더 아니면 디코더
	{
		bool rcv_wait = 0;
		while(1) {
			LOCK_MUT_(mutisle);
			if(syncurIndice < 0 || (isleimp->trainstep == 0 && isleimp->syncIndice != sync_indice)) {
				UNLOCK_MUT_(mutisle);//ieie.종료이거나 평가 단계수행이면 모든isle들이 동일 싱크로 실행되므로 이 값이  
				return;//평가시작할때 값과 틀리면 이전 학습단계의 플럭스가 전개 중간에 남아있던 것들이므로 제거한다.
			}
			if(islebusy || rcv_wait) {//isle이 수행중이면 대기
				cntwait++;
				UNLOCK_MUT_(mutisle);
				//printf("111-1: %p %d\n", this, cntwait);
				WAIT_USEM_(semijec);
				//printf("111-2: %p\n", this);
				if(rcv_wait == 0 && islebusy) {
					printf("in ject conflict\n");
					exit(1);
				}
				rcv_wait = 0;
			} else {
				if(sync_indice < 0) {//종료
					islebusy = 1;
					syncurIndice = sync_indice;
					CHKSIG_ISLE(1);
					UNLOCK_MUT_(mutisle);
				} else if(i_socket >= 0) {//입력과 타겟 연결학습중이고 서로다른소스(isle)2개값을 받는 경우
					if(cntoper) {//두번째 플럭스(소켓) 수신
						if(sockIndice[i_socket]) {//이전에 먼저 주입된쪽이 중첩되어 또 주입됐으면
							if(syncurIndice < sync_indice) {//이번에 주입한 소켓이 더 최근이면
								if(alignoper) {//ieij.최초 양측 합류 이후이면 상대 소켓이 수신되어 ㄴ)이나 ㄷ)에서 
									rcv_wait = 1;//시글널 할때까지 대기하고 ㄷ)이면 이전것은 버려지는 것이고 
									UNLOCK_MUT_(mutisle); //ㄴ)이면 이번것이 수행된것
									continue;//ㄹ.다른 한쪽 소켓이 수신될때까지 대기하여 최초 입력단까지 블럭되게 한다.(중간 isle들이 모두 대기중이라면)
								} else {//최초 양측 합류 이전이면 대기없이 다음을 주입토록 리턴
									islein[i_socket]->feedf(in);//이전것은 버리고 이번것으로 대체
									syncurIndice = sync_indice;
								}
							}//else 일반적으로는 같은 소켓에 뒤에 도할하는 플럭스이므로 이번에 도달한 소켓이 먼저  
							//수신된 소켓보다 더 오래된 케이스는 없으나(있다해도 버려지는 것이 정상) ㅂ)의 경우 올수있고 이전것은 버려진것
							UNLOCK_MUT_(mutisle);
						} else {//입력/타겟 오퍼랜드중 먼저 도달한 소켓의 상대편 소켓
							if(alignoper == 0) alignoper = 1;//처음으로 양측이 합류됨을 표시, 이전까지는 어느
							//한쪽의 중간 isle이 학습중이어서 여기까지 도될될수없는 상태일수있으므로(그렇치않고 
							//상태 소켓쪽보다 실행이 느려서 늦게 도달한 경우에는 처음 몇번만 한측만 중복해서 도달할
							//경우 버려지면 됨) 버리고 합류 이후부터는 계속 여기까지 도달해야 하므로 어느 한쪽이 
							//스텝이 많아 실행이 늦을 경우 대기하여 수행한다.(대기하면 입력단까지 블럭 걸릴수있고(
							//이쪽 소켓 패스의 모든isle이 대기중이라면) 이렇게하여 플로우 컨트롤 된다.
							if(syncurIndice == sync_indice) {//정상접수 수행스텝. 두개 소켓중 두번째로 수신된
								islein[i_socket]->feedf(in);//소켓의 스템프값을 비교하여 같으면 로드하고 
								islebusy = 1;				//학습 수행한다.
								cntoper = 0;//리셋하여 다음 입력/타겟 쌍 수신 준비
								sockIndice[0] = sockIndice[1] = 0;
								CHKSIG_ISLE(1);//ㄴ.
							} else {//양측이 다른 스템프, 수신받은 두갯 소켓의 싱크스템프가 틀리면
								if(syncurIndice < sync_indice) {//ㅂ.이번에 도달한 소켓이 더 최근이면
									islein[i_socket]->feedf(in);//이전것은 버리고 이번것으로 대체
									syncurIndice = sync_indice;//다른 한쪽 소켓이 대기중이면
									sockIndice[i_socket] = 1;//먼저 도착을 이번 소켓으로 바꾸고
									sockIndice[!i_socket] = 0;//후행을 상대편으로 바꿔고 상대편이
									CHKSIG_ISLE(0);//ㄷ.이전것을 버리고 탈출하여 다음을 수신하도록 시그널
								}// else cntoper = 0;//이번에 도달한 소켓이 먼저 수신된 소켓보다 더 오래된 
							}						//것이면 이번것은 버리고 다음 배치수행토록 리턴
							UNLOCK_MUT_(mutisle);
						}
					} else {//이전 학습이 수행된후 초기상태로서 압력/타겟 두개중 어느 하나 처음으로 receive이면  
						islein[i_socket]->feedf(in);//플럭스 적재후 계속 다음 배치를 수행하도록 리턴
						syncurIndice = sync_indice;
						cntoper = 1;
						sockIndice[i_socket] = 1;//0번 입력값, 1번 타겟값 어느 것이 이번에 수신됐는지 표시
						UNLOCK_MUT_(mutisle);
					}
				} else {//ijed.정상접수 수행스텝. i_socket이 음수이면 오퍼랜드 갯수로서 -2면 2개 모두 동일 소스(isle)
					//에서 주입되는 깃이므로 소켓 구분없이 적재(자기구조 디코더 학습, 입력과 타겟 양측이 자기구조학습
					//없이 바로 주입값을 연결하는 학습), 양수값이면 에측 수행일때로서 타겟값이 필요하지 않아 스펨프
					//비교하것이없어 수신 받으면 바로 수행
					islein[0]->feedf(in);
					if(i_socket == -2) islein[1]->feedf(tar);
					syncurIndice = sync_indice;//커플 isle 한곳에서 수신되므로 소켓은 1개만 사용되므로
					islebusy = 1;				//sockIndice표시 필요없다.
					CHKSIG_ISLE(1);
					UNLOCK_MUT_(mutisle);
				}
				return;//ㄹ)의 경우를 빼곤 대기없이 모두 리턴하여 다음을 수신.
			}
		}
	}
	bool peekevent(void)
	{
		LOCK_MUT_(mutisle);
		islebusy = 0;
		if(cntwait) {
			//printf("222- sig: %p %d\n", this, cntwait);
			SIG_USEM_(semijec, cntwait);
			cntwait = 0;
		}
		UNLOCK_MUT_(mutisle);
		//printf("222-1: %p\n", this);
		WAIT_USEM_(sempeek);
		//printf("222-2: %p\n", this);
		if(syncurIndice < 0) return 1;//아래 처음 한번은 async 메세지 출력 되고 이는 정상이다
		if(rcvsynIndice + 1 != syncurIndice) printf("async %d\n", (intt)(syncurIndice - rcvsynIndice));
		rcvsynIndice = syncurIndice;

		return 0;
	}
	void runisle(void)
	{
		while(1) {
			if(peekevent()) break;
			if(isleimp->trainstep > 0 || (isleimp->trainstep < 0 && fintrain == 0)) {//iril.1단계 
				trainloop(rcvsynIndice);//학습이거나 2단계 학습이면 디코더만 학습되는데 2단계 학습중인 
				isleimp->sigtraining();//디코더(아직 2단계 학습 완료되지 않은 디코더)만 학습 수행.
			} else predloop(rcvsynIndice);
		}
	}
};
class Encodeisle;
extern void convert_2d_to_1d(Flux *sor, intt dims[]);
//그리드 단위로 복원할 각 그리드를 감싸는 8개 압축코드 그리드를 입력 시퀀스로로 하고 복원할 한개 원본 그리드를
//타겟 시퀀스로 하여 연결학습한다
class Deocdeisle : public Isleloop {
public:
	bool finalDenseDec;//최종 인코더의 디코더
	Flux *originDout;//최조 인코더의 디코더
	Encodeisle *encisle;
	void buildislet(Flux *in, Flux *tar)
	{
		intt dims[MX_DIM];

		if(isletrc->printBuilding) {
			printf("decord isle in: ");
			in->shape();
			printf("decord isle tar: ");
			tar->shape();
		}
		finalDenseDec = 0;
		originDout = nullx;

		convert_2d_to_1d(in, dims);
		islein[0] = flux(isletrc, 3, dims, in->qType, variable);

		convert_2d_to_1d(tar, dims);
		islein[1] = flux(isletrc, 3, dims, tar->qType, variable);

		islenet = new(isletrc)Generic;
		Generic *conet = (Generic *)islenet;

		conet->makeAnet(isletrc, szlatent, actcode);
		conet->setFrontendFactor2(islein[0]);
		//encmhead();
		conet->decompose(islein[1]->fshape[1]);//시퀀스 길이가 크므로 decompose2로 할수 없다. 어텐션은 수행될수있음
		conet->connect(islein[1], headisle == HOD ? isoutdisc : isindisc, 0, rLeaning, isletrc->optType);//실제로는 디코더는 타겟 인코딩에서만 생성되므로 outdiscrt만 의미있으므로 조건 필요없으나 그냥 체크
		isleout = conet->cypred;
	}
	void encmhead(void);
	Deocdeisle(Impulse *imp, Dynagen *dgen, Encodeisle *encnet, Flux *in_gate, Flux *tar_gate, sytet head, bytet *name) : Isleloop(imp, dgen, head, name)
	{
		isleType = DECODE_ISLE;
		if(isletrc->frozenwgt < 2) fintrain = 0;//frozenwgt가 2이상이면 디코더 가중치 동결하도록 학습을 하지않게한다.
		imp->ndecisle++;
		encisle = encnet;
		//디코더는 외부에서 바로 연결되지 않으므로 extingate 설정 필요없다.
		buildislet(in_gate, tar_gate);
		noperand = 2;
	}
	void adaptOrigin(Flux *src)
	{
		//anet의 시퀀스는 2d가 없고 1차원으로만 수행하므로 입력이 2d일경우
		//입력과 모양이 같게 출력하기위해 입력 모양의 플럭스를 준비하고 출력을 여기 복사한다.
		if(src->fdim > 3) originDout = flux(isletrc, src, variable);
		else originDout = nullx;
	}
	Flux *restoreOrigin(Flux *dout)
	{
		if(originDout) {
			originDout->copyf(dout);
			return originDout;
		} else return dout;
	}
	void trainloop(longx sync_indice)
	{
		if(fintrain) return;
		//if(isleimp->trainstep < 0) printisle(islein[0], "decoder train in");
		Flux *r = trainislet(islein[0], islein[1]);//학습기간중에 학습끝나고 예측을 필요로하지 않으므로 학습만한다.
		lossout(r, "_train");//디코더의 학습이 종료 체크를 인코더에서하고 학습이 끝나면 eedp)에서 더이상 
					//디코더 학습으로 보내지않으므로 여기서 학습 종료 체크및 로스아웃 여부를 체크할 필요없다.
		finaltrain(r);
		if(isleimp->trainstep < 0 && isletrc->concurtrain) predloop(sync_indice);//2단계 학습
		//수행이고 동시학습이면 현 단계 학습후 하위후행 단계 디코더 학습위해 현 단계 평가 및 포워드
	}
	//void predloop(longx sync_indice) //디버그 용도
	//{
	//	Isleloop::predloop(sync_indice);
	//}
};

class QSortAllocator : public SortAllocator {
public:
	Trace *tracealt;

	QSortAllocator(Trace *trc)
	{
		tracealt = trc;
	}
	void *sortalloc(size_t size)
	{
		return tracealt->bxalloc(size);
	}
};
class Deconflict : public Typer {
public:
	intt xDiff, yDiff, xGrid, yGrid, xnGrid, ynGrid, xStride, yStride, xnStride, ynStride;
	intt xszZcode, ynZcode, bszGrid, szGrid, nScoop;//배치당 커널 스트라이드로 인해 파생되는 시퀀스 갯수
	intt szZcode, szFeat, xszReduce, szReduce;
	bool deconf2d;
	intt *mGrid;//reduced map
	virtual void checkinDeconflict(intt iorderive) = 0;
	virtual void resizem(void *zcodem, void *reductm, void *indicem) = 0;
	virtual void resizcode(void *zcodem) = 0;
};

template<typename DT>
class Deconflict2d : public Deconflict {
public:
	DT *mZcode;
	DT *mReduout;//indice map에 xy code size가 곱해진 맵
	Deconflict2d(void *zcodem, void *reductm, void *indicem, intt ydiff, intt xdiff, intt gridy, intt gridx,
		intt n_grid_y, intt n_grid_x, intt stridey, intt stridex, intt n_pred_scoop_y, 
		intt n_pred_scoop_x, intt ylen_zcode, intt xlen_zcode, intt feat_sz)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
		szFeat = feat_sz;
		yDiff = ydiff; xDiff = xdiff;
		yGrid = gridy; xGrid = gridx;
		ynGrid = n_grid_y; xnGrid = n_grid_x;
		yStride = stridey; xStride = stridex;
		ynStride = n_pred_scoop_y;
		xnStride = n_pred_scoop_x;
		ynZcode = ylen_zcode; 
		xszZcode = xlen_zcode * szFeat;
		nScoop = ynStride * xnStride;
		szGrid = ynGrid * xnGrid * 2;//y, x좌표
		szZcode = ynZcode * xszZcode;
		xszReduce = xnGrid * xszZcode;
		szReduce = ynGrid * ynZcode * xszReduce;//그리드 각 칸에 코드가 적재되는 사이즈
		deconf2d = 1;
	}
	void resizem(void *zcodem, void *reductm, void *indicem)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
	}
	void resizcode(void *zcodem)
	{
		mZcode = (DT *)zcodem;
	}
#define PGRID2d(deconf, pgrid, gy, gx) (pgrid + (gy * deconf->xnGrid + gx) * 2) //y, x좌표 두개씩
#define PZCODE(deconf, mzcode, ips) (mzcode + ips * deconf->szZcode)
#define PREDUCE2d(deconf, predu, gy, gx) (predu + gy * deconf->ynZcode * deconf->xszReduce + (gx * deconf->xszZcode))
	void checkinDeconflict(intt ipred_scoop)
	{
		intt ibatch = ipred_scoop / nScoop;
		intt ipred_rest = ipred_scoop - (ibatch * nScoop);
		intt y = (ipred_rest / xnStride) * yStride, x = (ipred_rest % xnStride) * xStride;//입력 앱상의 scoop시퀀스의 오리진 좌표x,y
		intt gy = y / yGrid, gx = x / xGrid;//grid matrix x,y좌표
		intt *pgrid = mGrid + ibatch * szGrid;//현 배치의 그리드 시작 포인터
		DT *pr = mReduout + ibatch * szReduce;//현 배치의 출력 축소 매트릑스 시작 포인터

		if(*PGRID2d(this, pgrid, gy, gx) >= 0) return;//이번 그리드가 같은 그리드 범위에 있는 것들중 이미 놓은  
										//정확도를 갖는 스트라이드 시퀀스 압축코드가 적재됏으면 이번것을 버린다.
		intt *gp;
		if(gy > 0) {//현 스트라이드 시퀀스 압축의 바로위 가 적재되있고 오버랩 범위내에 간섭되면 이번것들 버린다.
			gp = PGRID2d(this, pgrid, (gy - 1), gx);
			if(*gp >= 0 && y - *gp < yDiff) return;
		}
		if(gy < ynGrid - 1) {//하 방향 간접체크
			gp = PGRID2d(this, pgrid, (gy + 1), gx);
			if(*gp >= 0 && *gp - y < yDiff) return;
		}
		if(gx > 0) {//좌 방향 간접체크
			gp = PGRID2d(this, pgrid, gy, (gx - 1)) +1;//x좌표는 두번째이므로 +1
			if(*gp >= 0 && x - *gp < xDiff) return;
		}
		if(gx < xnGrid - 1) {//우 방향 간접체크
			gp = PGRID2d(this, pgrid, gy, (gx + 1)) +1;//x좌표는 두번째이므로 +1
			if(*gp >= 0 && *gp - x < xDiff) return;
		}
		gp = PGRID2d(this, pgrid, gy, gx);//현 그리드 좌표를 그리드 맵에 적재
		*gp = y; *(gp + 1) = x;
		//축소 출력 매트릭스에 이번 그리드 압축 코드 적재
		DT *rp = PREDUCE2d(this, pr, gy, gx), *zp = PZCODE(this, mZcode, ipred_scoop);
		for(intt i = 0;i < ynZcode; i++) {
			for(intt j = 0;j < xszZcode; j++) *(rp + j) = *zp++;
			rp += xszReduce;
		}
	}
};
template<typename DT>
class Deconflict1d : public Deconflict {
public:
	DT *mZcode;
	DT *mReduout;
	Deconflict1d(void *zcodem, void *reductm, void *indicem, intt xdiff, intt gridx, intt n_grid_x,
		intt stridex, intt n_pred_scoop_x, intt xlen_zcode, intt feat_sz)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
		szFeat = feat_sz;
		xDiff = xdiff; xGrid = gridx; szGrid = xnGrid = n_grid_x;
		xStride = stridex; xnStride = nScoop = n_pred_scoop_x;
		szZcode = xszZcode = xlen_zcode * szFeat;
		szReduce = xnGrid * xszZcode;
		deconf2d = 0;
	}
	void resizem(void *zcodem, void *reductm, void *indicem)
	{
		mZcode = (DT *)zcodem; mReduout = (DT *)reductm; mGrid = (intt *)indicem;
	}
	void resizcode(void *zcodem) 
	{
		mZcode = (DT *)zcodem;
	}
#define PGRID1d(deconf, pgrid, gx) (pgrid + gx)
#define PREDUCE1d(deconf, predu, gx) (predu + gx * deconf->xszZcode)
	void checkinDeconflict(intt ipred_scoop)
	{
		intt ibatch = ipred_scoop / xnStride;
		intt ipred_rest = ipred_scoop - (ibatch * xnStride);
		intt x = ipred_rest * xStride;//입력좌표x
		intt gx = x / xGrid;//grid matrix x좌표
		intt *pgrid = mGrid + ibatch * xnGrid;//현 배치의 그리드 시작 포인터
		DT *pr = mReduout + ibatch * szReduce;//현 배치의 출력 축소 매트릑스 시작 포인터

		if(*PGRID1d(this, pgrid, gx) >= 0) return;//이번 그리드가 같은 그리드 범위에 있는 것들중 이미 놓은 
										//정확도를 갖는  스트라이드 시퀀스 압축코드가 적재됏으면 이번것을 버린다.
		intt *gp;
		if(gx > 0) {//좌 방향 간접체크
			gp = PGRID1d(this, pgrid, gx - 1);
			if(*gp >= 0 && x - *gp < xDiff) return;
		}
		if(gx < xnGrid - 1) {//우 방향 간접체크
			gp = PGRID1d(this, pgrid, gx + 1);
			if(*gp >= 0 && *gp - x < xDiff) return;
		}
		*PGRID1d(this, pgrid, gx) = x;//현 그리드 좌표를 그리드 맵에 적재
		//축소 출력 매트릭스에 이번 그리드 압축 코드 적재
		DT *rp = PREDUCE1d(this, pr, gx), *zp = PZCODE(this, mZcode, ipred_scoop);
		for(intt j = 0;j < xszZcode; j++) *(rp + j) = *zp++;
	}
};
template<typename DT> class TeleReduction : public TeleSort {
public:
	Deconflict *inChecker;
	DT *ordermap;
	intx nSortRange, ibatchSort;

	TeleReduction(ParallelSort *ps, BaseSort *bs, Deconflict *deconf, intt nsort) : TeleSort(ps, bs)
	{
		inChecker = deconf;
		nSortRange = nsort;
	}
	void settings(intx s, intx e, void *dc, void *sc = nullx)
	{
		ibatchSort = s; 
		ordermap = (DT *)dc;
	}
	void sorting(void)
	{
		DT *porder = ordermap + ibatchSort * nSortRange * 2 + 1;//2(손실오차와 압축 코스 순번 인덱스) + 1(인덱스 기점 설정)
		for(intt i = 0;i < nSortRange; i++) {
			inChecker->checkinDeconflict((intt)*(porder + i * 2));
		}
	}
};
template<typename DT> class BaseReduction : public BaseSort {
public:
	Deconflict *inChecker;
	intx nSortRange;

	BaseReduction(ParallelSort *psort, SortAllocator *alc, Deconflict *deconf, intt nsort) : BaseSort(psort, alc)
	{
		inChecker = deconf;
		nSortRange = nsort;
	}
	TeleSort *newTeles(void)
	{
		TeleSort *ts = new(allocators, sortProsBs)TeleReduction<DT>(sortProsBs, this, inChecker, nSortRange);
		return ts;
	}
};
extern intt scoopeout_size(bool scoop_inner, intt seqy, intt seqx, intt slidey, intt slidex, intt stridey, intt stridex, intt &outy, intt &outx);
class Encodeisle : public Isleloop {
public:
	bool findectrain;
	intt zcompsize, cntDecTStep, szsortm, szindice, nzcDerive;
	void *outindice, *sortmap;
	Generic *conet;
	Deocdeisle *decisle;//predict일때는 최종 islepair만 decodisle로 eject한다.
	Deconflict *deconflict;
	QSortAllocator *qsrtalc;
	BaseSort *qsrtbase;
	BaseSort *redubase;
	void buildislet(Flux *in, intt &bound_sz)
	{
		isletrc->batchloss = 1;
		Flux *isle_in = islein[0] = in->duplicate(isletrc);//각 isle은 각각 자신의 tracer를 가지고 
		//오류역잔파가 독립적으로 수행되므로 입출력 플럭스들을 복사해야 한다. 안그러면 트레이스간에 그라프가 연결된다.
		bool d2 = isle_in->fdim > 3 ? 1 : 0;//슬라이드 커널은 정방형을 가정한다.
		intt seqy = d2 ? isle_in->fshape[1] : 0, seqx = d2 ? isle_in->fshape[2] : isle_in->fshape[1];
		intt sz_slide = (d2 ? std::sqrt(isletrc->szslide2d) : isletrc->szslide1d);//szslide2d는 1차원 기준이므로 
		intt sz_outrow, sz_outcol, out_grid = sz_slide;//출력간격
		intt diff = sz_slide - isletrc->overlapSlide;//겹쳐기 허용 간격
		intt out_shape[MX_DIM];//, ind_shape[MX_DIM];
		//슬라이드 사이즈, 학습 스트라이드로 시퀀스 추룰
		scoopout = isle_in->scoopup(sz_slide, sz_slide, isletrc->slTrainInterv, isletrc->slTrainInterv);//빌드
		intt discrete = (headisle == HID ? isindisc : isoutdisc);//현 isle이 입력/출력 인코딩의 끝단 isle일때 적용
		islenet = generic(isletrc, scoopout, scoopout, szlatent, discrete, discrete,
					szembede, actcode, rLeaning);//커널 자기구성학습 망
		conet = (Generic *)islenet;
		intt zcode_len = conet->zcodec->fshape[1];//디버깅때 사이즈 체크
		intt feat_sz = conet->zcodec->fshape[2];
		intt ylen_grid, xlen_grid, ind_sz;
		//오버랩 허용 간격으로 스트라이드 했을 경우 축소되는 그리드 매트릭스의 y,x 사이즈 계산
		scoopeout_size(1, seqy, seqx, sz_slide, sz_slide, out_grid, out_grid, ylen_grid, xlen_grid);
		if(d2) {
			zcode_len = std::sqrt(zcode_len) < 1 ? 1 : std::sqrt(zcode_len);//무조건 정방형으로 한다.
			//ind_shape[1] = ylen_grid;//reduce y축소매트릭스 채우기위한 간섭체크에 필요한 인덱스 맵 사이즈 설정.
			//ind_shape[2] = xlen_grid;//reduce x
			//ind_shape[3] = 2;//y,x 인덱스 두개
			szindice = ylen_grid * xlen_grid * 2 * sizeof(intt);//x,y 좌표 두개
			//커널 압축 코드를 정방형으로 만들 것이므로 x와 y에 같은 코드 길이를 곱한다. 예)커널 사이즈 8로 하여
			//scoop된 시퀀스가 16이면 16 / 8 == 2개 코드에 코드당 길이 2, 2 * 2 == 4
			//scoop된 시퀀스가 64이면 64 / 8 == 8개 코드에 코드당 길이 2, 8 * 2 == 16 -> y,x에 각각 4를 곱
			out_shape[1] = ylen_grid * zcode_len;//y,정방형으로 하기때문에 y축도 출력시퀀스 코드 사이즈를 곱한다
			out_shape[2] = xlen_grid * zcode_len;//x 커널당 출력시퀀스(1차원) 코드 사이즈를 곱한다.
			out_shape[3] = feat_sz;//피쳐 차원
			if(out_shape[1] * out_shape[2] < bound_sz) bound_sz = -1;//출력 사이즈가 일정 이하면 다음부턴 fully 연결
		} else {
			//ind_shape[1] = xlen_grid;//reduce x 인덱스
			//ind_shape[2] = 1;//x 인덱스 한개
			szindice = xlen_grid * sizeof(intt);//x좌표 한개
			out_shape[1] = xlen_grid * zcode_len;//커널당 출력시퀀스(1차원) 사이즈를 곱한다. 출력길이가 2개
			//이상이라도 상위에서 1개일때와 동일하게 각각을 1개 입력과 마찬가리로 독립적으로 처리한다. 출력 길이가 
			//2개이상 출력될때 각 출력 별로 개별 가중치가 곱해지는 것이므로 2개 값이 일반적인 자리수 생각하듯 
			//두개가 함께취급될때만 의미 있는 것이 아니므로
			out_shape[2] = feat_sz;//피쳐 차원
			if(out_shape[1] < bound_sz) bound_sz = -1;
		}
		out_shape[0] = isle_in->fshape[0];//배치 갯수 설정.
		isleout = flux(isletrc, isle_in->fdim, out_shape, isle_in->qType, variable);//출력 코드로 축소된 맵
		//outindice = flux(isletrc, isle_in->fdim, ind_shape, tint, variable);//인덱스 맵
		intt n_pred_scoop_y, n_pred_scoop_x;//예측때 간격으로 스트라이드 했을 때의 y,x 사이즈 계산 설정.
		nzcDerive = scoopeout_size(0, seqy, seqx, sz_slide, sz_slide, isletrc->slPredInterv, isletrc->slPredInterv, n_pred_scoop_y, n_pred_scoop_x);
		qsrtalc =  new QSortAllocator(isletrc);//소팅 할당자 생성.
		switch(isle_in->qType) {
			case NONET_TP:
				break;
			case BYTET_TP:
				break;
			case tshort:
				break;
			case tfloat:
				szsortm = sizeof(floatt) * 2;//오차와 압축 코드 순번 인덱스 pair 사이즈
				outindice = malloc(szindice * out_shape[0]);//인덱스 그리드 맵 할당.
				sortmap = malloc(szsortm * nzcDerive);//오차와 압축 코드 순번 인덱스 적재 공간
				if(d2) deconflict = new(isleimp->impTcr)Deconflict2d<floatt>(conet->zcodec->begin_p(),
					isleout->begin_p(), outindice, diff, diff, out_grid, out_grid,
					ylen_grid, xlen_grid, isletrc->slPredInterv, isletrc->slPredInterv,
					n_pred_scoop_y, n_pred_scoop_x, zcode_len, zcode_len, feat_sz);
				else deconflict = new(isleimp->impTcr)Deconflict1d<floatt>(conet->zcodec->begin_p(),
					isleout->begin_p(), outindice, diff, out_grid, xlen_grid,
					isletrc->slPredInterv, n_pred_scoop_x, zcode_len, feat_sz);
				qsrtbase = new BaseQSort<floatt>(rsc::cydrome->parallels, qsrtalc);
				redubase = new BaseReduction<floatt>(rsc::cydrome->parallels, qsrtalc, deconflict, nzcDerive);
				break;
			case tint:
				break;
			case tlong:
				break;
			case tdouble:
				break;
		}
	}
	Encodeisle(Impulse *imp, Dynagen *dgen, Flux *in_gate, intt &bound_sz, bool pair, sytet head, bytet *name, floatt lr = -1) : Isleloop(imp, dgen, head, name)
	{
		if(isletrc->printBuilding) {
			printf("encord isle: ");
			in_gate->shape();
		}
		isleType = ENCODE_ISLE;
		extingate = in_gate;//외부 게이트 설정
		cntDecTStep = 0;
		noperand = 1;
		if(lr > 0) rLeaning = lr;
		buildislet(in_gate, bound_sz);
		if(isletrc->printBuilding) {
			printf("encord isle out: ");
			isleout->shape();
		}
		if(pair) {
			strcat(name, "_decoder");
			decisle = new(imp->impTcr)Deocdeisle(imp, dgen, this, isleout, in_gate, head, name);
			//regiRouteTo(decisle, -2);eepl)에서 직접 주입하므로 라우팅 등록 필요없다.
			findectrain = 0;
		} else {
			decisle = nullx;
			findectrain = 1;
		}
	}
	void rmloop()
	{
		free(outindice);
		free(sortmap);
		delete qsrtalc;
		Isleloop::rmloop();
	}
	void trainloop(longx sync_indice)
	{
		if(fintrain) {
			predloop(sync_indice);
			return;
		}
		scoopout->scoopup(isletrc->slTrainInterv, isletrc->slTrainInterv);
		Flux *r = trainislet(scoopout, scoopout);//scoopout은 빌드할때 입력으로 설정되있으므로 입력은 의미없이 그냥한다.
		finaltrain(r);
		lossout(r, "_train");
		if(isletrc->concurtrain) predloop(sync_indice);//동시학습이면 현 단계 학습후 상위 단계 학습위해 현 단계 평가 및 포워드
	}
	void predloop(longx sync_indice)
	{
		Flux *in = islein[0];
		scoopout->scoopup(isletrc->slPredInterv, isletrc->slPredInterv);//scoopout에 scoop된 결과 적재됨
		predislet(scoopout, &predloss);//scoopout은 빌드할때 입력으로 설정되있으므로 입력은 의미없이 그냥한다.
		//1.inner scoop - 스트라이드와 오버랩 출력 구성을 별개로 생각한다. 스트라이드는 이 간격으로 피쳐맵을 발췌하는 것이고
		//오버랩은 이렇게 발췌한 피쳐맵들로 출력 맵를 구성할때 커널사이즈에서 몇간을 인접 피쳐들간에 겹치기 허용할 것이냐 이다
		//따라서 출력 맵의 사이즈는 커널사이즈에서 오버랩 사이즈를 뺀 사이즈로 입력 맵을 나눈 몫[(n + d -1) / d]이 된다.
		//예로 커널사이즈 4*4, 입력 사이즈 32*32라면 오버랩 없으면 32 + (4-1) / 4 = 8, 8*8이 되고 1칸 오버랩 허용하면
		//32 + (3-1) / 3 = 11, 11*11이 된다. 이 출력 맵에 스트라이드하여 발췌한 4*4 피쳐맵의 쪽 상위 모서리를 오리진으로
		//하여 입력맵에서의 좌표를 전자는 4로 나눈몫, 후자(오버랩1)는 3으로 나눈 몫을 좌표로 그룹핑(나머지가 남는 것들)하여 
		//그룹된 피쳐들 간에 오차 낮은 순으로 출력맵에 인덱싱하고 오차 낮은순 후순위가 선순위가 이미 좌표선점되있어 중복된 경우와
		//출력맵상에서 이번에 인덱싱할 좌표의 상하좌우 네곳에 먼저 인덱싱된 것들이 있으면 이의 입력맴상의 좌표로 비교하여 
		//허용 오버랩을 넘어서면 이번 피쳐맵을 버린다. 스트라이드가 한칸이 아닌 두칸, 세칸 간격이러면 이 간격으로 피쳐맵을 
		//발췌하여 위 방법과 동일하게 매핑한다. 스트라이드 두카이상이면 위 출력맵 구성에서 빈곳이 생기고 이는 어쩔수없다. 
		//커널사이즈를 32로하면 4개 시퀀스를 얻를수있고 ORTHO_DOT로 연산했으면 4개 시퀀스를 한 덩어리로 볼수있으므로 중간레벨부터는 
		//스트라이드2칸에 오버랩 혀용 2칸으로 하여 4개 시퀀스를 한덩어리로 취급할수있다. ORTHO_DOT을 ortho_single로
		//실행한다면 4개시퀀스 각각을 개별로 최급하여 스트라이드나 오버랩할때 1칸씩 해도 된다.(확정적인 것은 아님)
		//예측할때는 1칸 간격으로 스트라이드하고 학습 할때는 넓은 간격으로 스트라이드 한다.
		//분류,검색등의 앱에서는 일정 오차 이상 피쳐들은 버림하고(컨브넷의 맥스풀링가 같은 의미) ORTHO_DOT을 ortho_single로
		//실행하여 시퀀스를 없앰으로서 필터링 및 예측 성능을 좋게한다. 사이즈 차이의 문제는 최대 사이즈 이미지로 학습해 놓고
		//작은 사이즈 이미지는 큰사이즈 이미지보다 좀더 상위 레렐에서 비교를 시작하여 상위로 진행하는 것 검토, 
		//오토인코딩 방식이므로 오차를 측정하여 적절한 레벨을 찾을수 있을 것

		//2.outer scoop - 마지막 커널 사이즈 구간에서 오버랩(스트라이드)에 의해 파생되는 피쳐들은 전체 시퀀스 사이즈를
		//넘는 것은 제로패딩되므로 항상 맨 뒤의 피쳐가 제일 정확도가 높게 되므로 위 사이즈를 구하는 식을 
		//[((n - k) + d -1) / d] + 1 와 같이 약간 변형한다. k는 커널 사이즈
		//예) n: 8, k: 4, (1) d: 4 -> [(4 + 3) / 4] +1 => 2, (2) d: 3 -> [(4 + 2) / 3] +1 => 3, 
		//(3) d: 2 -> [(4 + 1) / 2] +1 => 3, (4) d: 1 -> [(4 + 0) / 1] +1 => 5
		//(3)번 같은 경우 원래 집합은 [0,2,4,6]이지만 4와 6은 마지막 커널구간 [4 - 8]에서 [4,5,6,7]과 [6,7,0,0]과 같이
		//두개 피쳐가 파생되고 두번째 피쳐에서 끝에 제로패딩 2개가 추가되어 제로패딩은 한상 같으므로 왠만하면 두번째 피쳐가 
		//정확도가 높게 나올것이므로 왜곡 될수있음므로 두번째 식으로 한다.

		//3. 2.번과 같이 할경우 스트라이드를 1로 오버라이드를 1로 했을때 오버라이드 그리드는 [0,1,2,3] [3,4,5,6] [6,7,0,0]과
		//같이되고 스트라이드 그리드는 [0,1,2,3] ~ [4,5,6,7]로 되고 오버라이드 그리드 [6,7,0,0]에 속하는 파생 시퀀스가
		//스트라이드 그리드에서 생산되질 않는다. 따라서 2.번과 같이 [(n + d -1) / d]로 한다. 2.번 이유에서 [6,7]이 항상
		//끝에와서 그것의 정확도가 높다면 이것이 선택되는 것이 맞고 그 앞쪽은 [3,4,5,6]이 선택될수있으므로 괜찬다.
		//맨앞쪽에 아웃터 스쿱을 안하는 것은 그 뒤에서 제일 앞쪽이 [3,4,5,6]인데 이것이 선택된다 해도 [0,1,2,3]이 선택되면
		//하나도 누실되는 것이 없으므로 앞쏙에는 아웃터 스쿱을 할 필요없고 [0,1,2,3] 안에서 [0,1]이 더 정확하던 [2,3]이 더 
		//정확하던 일단 [0,1,2,3]이 패턴으로 입력되면 완전 조합으로 학습되므로 상관없다.
		intt n_batch = in->fshape[0];
		if(isleout->fshape[0] != n_batch) {
			isleout->resizing2(in);
			free(outindice);
			free(sortmap);
			outindice = malloc(szindice * n_batch);//인덱스 그리드 맵 재 할당.
			sortmap = malloc(szsortm * conet->zcodec->fshape[0]);//fshape[0]는 배치와 파생시퀀스 갯수가 곱해진 사이즈, 오차와 압축 코드 순번 인덱스 적재 공간 재 할당
			deconflict->resizem(conet->zcodec->begin_p(), isleout->begin_p(), outindice);
		} else deconflict->resizcode(conet->zcodec->begin_p());//gpu실행됐으면 내용을 호스트 메모리로 복사시키기고, 
																//재설정, isleout은 싫행없으므로 pass
		memset(outindice, -1, szindice * n_batch);
		isleout->resetData(-1);//호스트 메모리만 제로 리셋
		//conet->zcodec->printo(2, 10);
		//dbg_check_mark(conet->zcodec->begin_p(), predloss->begin_p(), n_batch, conet->zcodec->fshape[0], predloss->fshape[0]);//dbg_check2를 할며면 실행.
		TeleSort *sort_head = nullx, *reduct_head = nullx, *sts, *rts;
		void *loss_m = predloss->begin_p();
		intt i_beg = 0;
		//printf("aaa-1: %p\n", this);
		for(intt i = 0;i < n_batch; i++, i_beg += nzcDerive) {
			sts = qsrtbase->getTeles();
			APPEND_LIST2(sort_head, sts);
			sts->settings(i_beg, i_beg + nzcDerive, sortmap, loss_m);
			sts->rsort();//정렬 수행.
		}
		sts = sort_head;
		for(intt i = 0;i < n_batch; i++, sts = sts->ptrRight2) {
			sts->wsort();
			//dbg_print_sort(sortmap, loss_m, conet->zcodec->begin_p(), nzcDerive);
			//reductionOut(i);//병렬처리 안할려면 아래 대신에 타입에 때라 호출 
			rts = redubase->getTeles();
			APPEND_LIST2(reduct_head, rts);
			rts->settings(i, -1, sortmap, nullx);
			rts->rsort();//reduction 수행
		}
		//printf("aaa-2: %p\n", this);
		for(rts = reduct_head;rts; rts = rts->ptrRight2) {
			rts->wsort();
		}
		//printf("aaa-3: %p\n", this);
		//printisle(isleout, "encode pred");
		//dbg_check2(conet->zcodec->begin_p(), isleout->begin_p(), sortmap, predloss->begin_p(), n_batch, 1);//dbg_check_mark를 실행하면 끝을 1로, 아니면 0로 호출
		//doublet mincode, maxcode;
		//isleout->minmax(mincode, maxcode);
		//isleout->minmax_normal(mincode, maxcode);
		isleout->stdnormal();
		//printisle(isleout, "encode pred2");
		lossout(predloss, "_pred");
		//isleout eject나 inject에서 메모리 포인터 복사하는데만 사용되므로 호스트 메모리 설정 필요없다.
		if(isleimp->trainstep < 1 || findectrain) {//예측수행이거나 2단계 학습 혹은 1단계 학습인데
			Isleloop::eject(isleout, sync_indice);//디코더 학습까지 끝났으면 상위후행 학습으로 포워드
			if(decisle) {
				if(isleimp->trainstep > 0) isleimp->sigtraining();//1단계학습인데 디코더 학습이 끝났으면
				//다코더는 실행안하므로 동시학습 2단계 수행이면 디코더는 수행완료 카운팅만 한다.
				if(isleimp->predwithloss) decisle->islein[1]->feedf(in);//평가때 타겟이 주어저 로스계산이면 디코더 타겟 직접 주입
			}
		} else {//eedp.인코더의 학습이 끝났으면 디코더 학습 수행.(디코더가 없으면 여기로 오지 않는다)
			decisle->inject(isleout, in, sync_indice, -2);//eepl.디코더의 0번 소켓에 preds를 입략값으로, in을 타겟값으로 직접 주입
			Isleloop::eject(isleout, sync_indice);//상위로 포워드하여 상위 학습, 디코더와 선후관계가 없으므로 동시 학습됨.
		}
		//printf("aaa-3: %p\n", this);
	}
	template<typename DT> void reductionOut(intt ibatch)
	{
		DT *porder = (DT *)sortmap + ibatch * nzcDerive * 2 + 1;//2(손실오차와 압축 코스 순번 인덱스) + 1(인덱스 기점 설정)
		for(intt i = 0;i < nzcDerive; i++) {
			deconflict->checkinDeconflict((intt)*(porder + i * 2));
		}
	}
	void dbg_print_sort(void *_smap, void *_lmap, void *_z_map, intt ns)
	{
		Deconflict *df = deconflict;
		floatt *smap = (floatt *)_smap, *lmap = (floatt *)_lmap, *z_map = (floatt *)_z_map;
		floatt v = -100;//오름차순
		//floatt v = 100;//내림차순
		intt x, y, ips;

		LOCK_MUT_(isleimp->mutimp);
		for(intt i = 0;i < ns; i++) {
			ips = (intt)*(smap + i * 2 + 1);
			if(*(smap + i * 2) < v) {//오름차순
			//if(*(smap + i * 2) > v) {//내림차순
				printf("inverse: %f %d\n", v, ips);
				break;
			}
			y = (ips / df->xnStride) * df->yStride, x = (ips % df->xnStride) * df->xStride;
			if(*(smap + i * 2) == v) printf("dup: %f %d (%d, %d)\n", v, ips, y, x);
			else printf("%f %d (%d, %d)\n", *(smap + i * 2), ips, y, x);
			v = *(smap + i * 2);
			if(v != *(lmap + ips)) {
				printf("error %f\n", *(lmap + ips));
				break;
			}
			for(intt j = 0;j < df->szZcode; j++) printf("%f ", *(z_map + ips * df->szZcode + j));
			printf("\n");
		}
		UNLOCK_MUT_(isleimp->mutimp);
	}
	void dbg_check_mark(void *z_map, void *b_loss, intt n_batch, intt zc_batch, intt l_batch)
	{
		Deconflict *df = deconflict;
		intt x, y;

		if(n_batch * nzcDerive != zc_batch) exit(1);
		if(zc_batch != l_batch) exit(1);
		if(nzcDerive != deconflict->nScoop) exit(1);

		floatt *zp = (floatt *)z_map, *lp = (floatt *)b_loss, *ep;
		intt sz_feat = deconflict->szFeat;
		//타입은 floatt, 한개 z_map는 4개 길이로 고정, 출력된 압축코드 맵에 각 코드의 순번과 오차를 적재
		for(intt i = 0;i < n_batch; i++) {				//[index, 0
			for(intt j = 0;j < nzcDerive; j++, lp++) {	// 0, loss]와 같이 피쳐에 동일하게 정방형으로 설정.
				ep = zp + df->szZcode;
				if(df->deconf2d) {
					y = (j / df->xnStride) * df->yStride, x = (j % df->xnStride) * df->xStride;
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)y;
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)x;
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)j;
					for(intt k = 0;k < sz_feat; k++) *zp++ = *lp;
				} else {
					for(intt k = 0;k < sz_feat; k++) *zp++ = (floatt)j;
					if(df->xszZcode > 1) {
						for(intt k = 0;k < sz_feat; k++) *zp++ = *lp;
					}
				}
				for(;zp < ep; zp++) *zp = -1;
			}
		}
	}
	void dbg_check2(void *z_map, void *r_map, void *s_map, void *b_loss, intt n_batch, bool dgb_chk)
	{//위 함수를 실행했으면 dgb_chk를 1로 한다.
		floatt *sp = (floatt *)s_map, *lp = (floatt *)b_loss;
		floatt *rp = (floatt *)r_map, *rp3, *zp = (floatt *)z_map, *zp3;
		Deconflict *df = deconflict;

		intt sz_redu = df->szReduce, sz_feat = df->szFeat;
		intt xsz_redu = df->xszReduce, n_scoop = df->nScoop;
		intt x, y, gx, gy, *pg, *gp, ipred_scoop;

		for(intt i = 0;i < n_batch; i++, rp += sz_redu) {
			pg = df->mGrid + i * df->szGrid;//현 배치의 그리드 시작 포인터
			//rp - 현 배치의 출력 축소 매트릑스 시작 포인터
			if(df->deconf2d) {
				for(intt j = 0;j < n_scoop; j++) {//한개 배치내 파생 시퀀스를 시퀀스 단위로 순차적으로 체크
					y = (j / df->xnStride) * df->yStride;
					x = (j % df->xnStride) * df->xStride;//입력 앱상의 scoop시퀀스의 오리진 좌표x,y
					gy = y / df->yGrid, gx = x / df->xGrid;//grid matrix x,y좌표
					gp = PGRID2d(df, pg, gy, gx);//현 x,y좌표가 align된 그리드 좌표에 적재된 scoop시퀀스의 오리진
					if(*gp != y || *(gp + 1) != x) continue;//좌표x,y와 현 x,y좌표가 같지않으면(선택되지않은 파생 시퀀스는) 스킵
					rp3 = PREDUCE2d(df, rp, gy, gx);//현 파생시퀀스압축코드 j의 출력축소맵상의 시작포인터 			
					if(dgb_chk) {//dbg_check_mark에서 현 파생시퀀스의 sort맵에 적재된 순번과 오차가 축소맵에 재대로 적재됐는지 확인
						rp3 += xsz_redu;//[2,2]형태로 적재됐고 다음 로우에 순번과 오차.
						for(intt b = 0;b < sz_feat; b++) {//피쳐수만큼 중복 적재돼있음.
							if(*(rp3 + b) != j) {
								exit(1);//세번째에 순번이 적재된 것이 일치하는가 체크
							}
							if(*(rp3 + sz_feat + b) != *(lp + i * n_scoop + j)) {
								exit(1);//마지막(네번째)에 손실오차 적재된것 일치여부 체크
							}
						}
					} else {//파생시퀀스 압축코드 맵의 각 코드가 출력축소 맵에 제대로 적재됐나 검사
						ipred_scoop = i * n_scoop + ((y * df->xnStride) / df->yStride) + (x / df->xStride);
						if(ipred_scoop != i * n_scoop + j) {
							exit(1);//파생시퀀스 순번
						}
						zp3 = PZCODE(df, zp, ipred_scoop);
						for(intt a = 0;a < df->ynZcode; a++) {//[2,2]형태로 적재했으므로 2개로우 수행
							for(intt b = 0;b < df->xszZcode; b++) {
								if(*(rp3 + b) != *zp3++) {
									exit(1);
								}
							}
							rp3 += xsz_redu;
						}
					}
				}
			} else {
				for(intt j = 0;j < n_scoop; j++) {//한개 배치내 파생 시퀀스를 시퀀스 단위로 순차적으로 체크
					x = j * df->xStride;//입력 앱상의 scoop시퀀스의 오리진 좌표x
					gx = x / df->xGrid;//grid matrix x좌표
					gp = PGRID1d(df, pg, gx);//현 x좌표가 align된 그리드 좌표에 적재된 scoop시퀀스의 오리진
					if(*(gp + 1) != x) continue;//좌표x와 현 x좌표가 같지않으면(선택되지않은 파생 시퀀스는) 스킵
					rp3 = PREDUCE1d(df, rp, gx);//현 파생시퀀스압축코드 j의 출력축소맵상의 시작포인터 			
					if(dgb_chk) {//dbg_check_mark에서 현 파생시퀀스의 sort맵에 적재된 순번과 오차가 축소맵에 재대로 적재됐는지 확인
						for(intt b = 0;b < sz_feat; b++) {
							if(*(rp3 + b) != j) exit(1);//첫번째에 순번이 적재된 것이 일치하는가 체크
							if(df->xszZcode > 1) {
								if(*(rp3 + sz_feat + b) != *(lp + i * n_scoop + j)) {
									exit(1);//두번째에 손실오차 적재된것 일치여부 체크
								}
							}
						}
					} else {
						zp3 = PZCODE(df, zp, j);
						for(intt a = 0;a < df->xszZcode; a++) {
							if(*(rp3 + a) == *zp3++) exit(1);
						}
					}
				}
			}
		}
	}
};
class CoupleIsle : public Isleloop {
public:
	CoupleIsle(Impulse *imp, Dynagen *dgen, sytet head, bytet *name) : Isleloop(imp, dgen, head, name)
	{
		isleType = COUPLE_ISLE;
		if(isletrc->frozenwgt < 3) fintrain = 0;//frozenwgt가 3이상이면 커플isle 가중치 동결하도록 학습을 하지않게한다.
		nprotect = 2;//오러랜드가 입력 과 타겟 두개이므로 
	}
	void trainloop(longx sync_indice)
	{
		if(fintrain) {//커플isle의 학습이 끝났으면 포워드 할 것들중에 디코더가 아닌 것이 하나라도 있으면 포워드
			bool forward_train = 0;
			for(Socketisle *s = isleRoute;s; s = s->ptrRight) {
				if(s->destisle->isleType != DECODE_ISLE) forward_train = 1;
			}
			if(forward_train) predloop(sync_indice);//이 함수의 e ject에서 디코더로의 포워드는 2단계 학습 혹은 평가단계에서만 된다.
		} else {//커플isle 학습 수행, 이 함수안에서 동시학습 오션일 경우 pred호출되며 상위로 포워드 된다.
			//printf("=====================================\n");
			//islein[0]->printo();
			Isleloop::trainloop(sync_indice);//이 함수 e ject에서 학습 1단계에서는 디코더가 아닌 것들만 포워드된다.
		}
	}
	/*void predloop(longx sync_indice)
	{
		printf("=====================================\n");
		islein[0]->printo();
		Isleloop::predloop(sync_indice);
	}*/
};
class Coaxisle : public CoupleIsle {
public:
	Flux *dualbgate;

	void buildislet(Flux *in, Flux *tar)
	{
		intt dims[MX_DIM];

		convert_2d_to_1d(in, dims);
		islein[0] = flux(isletrc, 3, dims, in->qType, variable);

		convert_2d_to_1d(tar, dims);
		islein[1] = flux(isletrc, 3, dims, tar->qType, variable);
		
		if(isleimp->impTcr->bygate && szembede < 0) {//바이게이트가 설정됐고 classification에서
			//호출되었으면 듀얼 인코더로 실행한다. 바이케이트는 외부에서 직접 데이터가 로드되기때문에 
			//입력데이터와 싱크를 맞출수없으므로 학습을 동기식으로 수행한다.
			dualbgate = isletrc->migbygate(isleimp->impTcr);
			//setdgendual();
		}
		islenet = generic(islein[0], islein[1], szlatent, isindisc, isoutdisc, szembede, actcode, rLeaning);//csnr.
		isleout = ((Generic *)islenet)->cypred;
		//isleimp->accucell = islenet;//ipare.
		//isleimp->accutrc = isletrc;
	}
	void setdgendual(void);
	Coaxisle(Impulse *imp, Dynagen *dgen, Flux *in_gate, Flux *tar_gate, sytet head, bytet *name) : CoupleIsle(imp, dgen, head, name)
	{
		dualbgate = nullx;
		if(isletrc->printBuilding) {
			printf("coax isle in: ");
			in_gate->shape();
			printf("coax isle tar: ");
			tar_gate->shape();
		}
		extingate = in_gate;
		exttargate = tar_gate;
		setdgencoup(dgen);
		buildislet(in_gate, tar_gate);
		noperand = 2;
	}
};
class Stratusisle : public CoupleIsle {
public:
	void buildislet(Flux *in, Flux *tar)
	{
		intt dims[MX_DIM];

		convert_2d_to_1d(in, dims);//모먕만 바뀌고 사이즈는 동일, inject에서 복사 됨.
		islein[0] = flux(isletrc, 3, dims, in->qType, variable);

		convert_2d_to_1d(tar, dims);
		islein[1] = flux(isletrc, 3, dims, tar->qType, variable);

		islenet = stratus(islein[0], islein[1], szlatent, isindisc, isoutdisc, szembede, actcode, rLeaning, nullx);
		//isleimp->accucell = islenet;//ipare.
		//isleimp->accutrc = isletrc;
	}
	Flux *trainislet(Flux *in, Flux *tar)
	{
		return ((Stratus *)islenet)->train();
	}
	Flux *predislet(Flux *in, Flux **loss)
	{
		return ((Stratus *)islenet)->predict(loss);
	}
	Stratusisle(Impulse *imp, Dynagen *dgen, Flux *in_gate, Flux *tar_gate, sytet head, bytet *name) : CoupleIsle(imp, dgen, head, name)
	{
		if(isletrc->printBuilding) {
			printf("stratus isle in: ");
			in_gate->shape();
			printf("stratus isle tar: ");
			tar_gate->shape();
		}
		setdgencoup(dgen);
		buildislet(in_gate, tar_gate);
		noperand = 2;
	}
};
#define T_DG_TRANSLATE	0
#define T_DG_RECALL		1
#define T_DG_CALSS		2
#define T_DG_CONNECT	3
#define T_DG_ENCODE		4

//translate, recall, classification, connect
class Dynagen : public Typer {
public:
	Impulse *dgenimp;
	Isleloop *inCodeisle, *coupleisle;
	Encodeisle *tarCodeisle;
	intt inz_bound, tarz_bound, coupConvolv;
	intt szlatent, dgindisc, dgoutdisc, szembede;
	sytet actcode, lossIndice;
	floatt rLeaning, couplowbo;
	Flux *dgenloss[2], *dgenpred, *dgenbgate, *dumloss;
	lsemx semdgen;
	CydChain *cydgen;
	Dynagen *ptrLeft, *ptrRight;
	bytet dgenname[128];
	void dgeninst(Impulse *imp, const bytet *name)
	{
		intt rv;

		dgenimp = imp;
		imp->regidgen(this);

		cydgen = rsc::cydrome->getcydid();
		CRE_LSEM_(semdgen, 0, cydgen->sem_iden, rv);
		if(rv < 0) {
			throwFault(-1, (bytex *)"cre sem fail\n");
		}
		dgenpred = nullx;

		if(name) {
			if(strlen(name) >= 128) throwFault(-1, (bytex *)"name over fail\n");
			sprintf(dgenname, "%s_%d", imp->cntdgen++, name);
		} else sprintf(dgenname, "_dynagen_%d", imp->cntdgen++);
	}
	void dgenremove(void)
	{
		CLOSE_LSEM_(semdgen);
		rsc::cydrome->putcydid(cydgen);
	}
	void sigprec(Flux *pred)
	{
		dgenpred = pred;
		SIG_LSEM_(semdgen);
	}
	void waitprec(void)
	{
		WAIT_LSEM_(semdgen);
	}
	Flux *predictv(Flux **loss)
	{
		//printf("return loss: %p pred: %p\n", dgenloss[0], dgenpred);
		if(loss) *loss = dgenloss[0];//최종 커플 isle의 로스
		dgenloss[0] = dumloss;//다음에 dgenloss에 새로운 로스가 설정되기전에 호출되면 더미로스가 
		return dgenpred;		//다시 참조되도록 
	}
	void dumyloss(void)
	{
		intx dim[1];
		dim[0] = 1;
		dgenloss[0] = dumloss = flux(dgenimp->impTcr, 1, dim, tfloat, variable);
	}
	Flux *getbygate(void) { return dgenbgate; }
	Dynagen(sytet t_dynagen, Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{
		Isleloop *isl, *next;
		Isleloop *in_head_isle, *tar_head_isle;
		sytet tar_sid = -1;
		intt rsz, i;
		bytet _name[128];

		dgeninst(imp, name);

		szlatent = latent_sz;
		dgindisc = indiscret;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;

		inz_bound = imp->impTcr->inzbound;
		tarz_bound = imp->impTcr->tarzbound;
		in_head_isle = tar_head_isle = nullx;
		if(t_dynagen < T_DG_CALSS) {//translate, recall, 8길이까지 압축한다. 타겟 시퀀스 인코딩/디코딩 학습
			if(t_dynagen == T_DG_RECALL) {//리콜이면 입력케이트로 매개되나 타겟 변수로 통일시켜 처리, 예측할때 ingate로
				targate = ingate;			//입력하고 인케이트의 디코더로 결과를 받는다.
				tarz_bound = inz_bound;
			}
			rsz = tarz_bound;
			if(t_dynagen == T_DG_TRANSLATE) lossIndice = 1;//번역이면 타겟시퀀스 로스는 1번에 적재
			else lossIndice = 0;//번역이 아니면 타겟시퀀스 학습은 없으므로 로스는 0번에만 적재하면 된다.
			i = 0; sprintf(_name, "tar_encode_isle_%d", i++);
			for(tar_head_isle = isl = next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, targate, rsz, 1, HOD, _name); rsz > 0; isl = next) {
				sprintf(_name, "tar_encode_isle_%d", i++);
				next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, next->isleout, rsz, 1, NHNDT, _name);//디코더를 함계 빌드
				isl->regiRouteTo(next, -1);
				((Encodeisle *)next)->decisle->regiRouteTo(((Encodeisle *)isl)->decisle, -1);//decoder간 <-- 방향 연결
			}
			tarCodeisle = (Encodeisle *)next;//last encoder
			tarCodeisle->decisle->finalDenseDec = 1;
			((Encodeisle *)tar_head_isle)->decisle->setFinal(1);
			((Encodeisle *)tar_head_isle)->decisle->adaptOrigin(ingate);
			if(t_dynagen == T_DG_RECALL) {//최종 인코더의 출력을 디코더 입력으로 주입(예측을 위한 연결)
				tarCodeisle->regiRouteTo(tarCodeisle->decisle, -1);
			}
		}
		lossIndice = 0;//입력 시퀀스(ingate) isle및 커플 isle 로스는 0번에 적재
		if(t_dynagen == T_DG_TRANSLATE || t_dynagen == T_DG_CALSS || t_dynagen == T_DG_ENCODE) {
			rsz = inz_bound;
			i = 0; sprintf(_name, "in_encode_isle_%d", i++);
			for(in_head_isle = isl = next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, ingate, rsz, 0, HID, _name); rsz > 0; isl = next) {
				sprintf(_name, "in_encode_isle_%d", i++);
				next = new(dgenimp->impTcr) Encodeisle(dgenimp, this, next->isleout, rsz, 0, NHNDI, _name);//입력측은 재현이 필요없으므로 디코더 빌드 안한다.
				isl->regiRouteTo(next, -1);
			}
			inCodeisle = next;//last encoder
			coupConvolv = 4;
			couplowbo = 1;
			strcpy(_name, "couple_isle");
			if(t_dynagen == T_DG_TRANSLATE) {
				if(imp->impTcr->stratusOpt == 0) {
					coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, inCodeisle->isleout, tarCodeisle->isleout, NHNDC, _name);
				} else {
					coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, inCodeisle->isleout, tarCodeisle->isleout, NHNDC, _name);
				}
				inCodeisle->regiRouteTo(coupleisle, 0);//예측때는 입력 라우팅만으로 실행
				tarCodeisle->regiRouteTo(coupleisle, 1);//훈련때는 입력과 타겟 두개의 스템프가 일치해야 실행.
				coupleisle->regiRouteTo(tarCodeisle->decisle, -1);//커플 isle의 예측줄력을 타겟 최종isle디코더 입력으로 주입(예측을 위한 연결)
			} else if(t_dynagen == T_DG_CALSS) {//classification
				if(imp->impTcr->stracode == 0) {//Coaxisle의 입력이 압축코드이므로 -1 설정.듀얼인코더 설정일때 의미.
					szembede = -1;//이후 이 변수는 dgen에서는 더이상 사용되 않으므로 복원할 필요없다.
				}//else dncs.stracode이면 다이나젠에서 하위 인코딩하고 아래 제너릭에서 한번더 중간 인코딩후 
				//최종 듀얼인코더에 입력하게 하도록 szembede를 그대로 둔다.
				coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, inCodeisle->isleout, targate, NHOC, _name);
				dgenbgate = ((Coaxisle *)coupleisle)->dualbgate;
				inCodeisle->regiRouteTo(coupleisle, 0);//입력의 최종isle의 출력을 커플의 입력으로 주입
				tar_head_isle = coupleisle;//목표값을 커플isle의 목표값으로 바로 주입
				tar_sid = 1;//목표값으로 주입되야하므로 1.
				coupleisle->setFinal(1);
			} else inCodeisle->setFinal(1);//T_DG_ENCODE, 현재는 죄종설정, 후애 연결되면 리셋됨.
		}
		if(in_head_isle) dgenimp->regifeed(in_head_isle, -1);
		if(tar_head_isle) dgenimp->regifeed(tar_head_isle, tar_sid);//recall이면 입력 매개변수가 타겟 변수로 전용되어 등록된다.
	}
	//이하 입력으로 Isleloop는 최종단 이다.
	Dynagen(Impulse *imp, Isleloop *in_zc_isle, Isleloop *tar_zc_isle, bool pred_link = 1, 
		sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)//T_DG_CONNECT
	{
		dgeninst(imp, name);

		in_zc_isle->setFinal(0);//encoder일 경우 이번에 후행이 연결되므로 최종이 아니게 되어 리셋한다.
		tar_zc_isle->setFinal(0);
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, in_zc_isle->isleout, tar_zc_isle->isleout, NHNDC, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, in_zc_isle->isleout, tar_zc_isle->isleout, NHNDC, (bytet *)"copule_isle");
		}
		in_zc_isle->regiRouteTo(coupleisle, 0);//잠재코드 입력값을 커플isle의 입력값으로 주입
		tar_zc_isle->regiRouteTo(coupleisle, 1);//잠재코드 타겟값을 커플isle의 타겟값으로 주입
		if(pred_link) {//예측을 위한 연결, 커풀의 예측 출력값을 타겟isle의 디코더의 입력값으로 주입,이때 타겟isle은
			coupleisle->regiRouteTo(((Encodeisle *)tar_zc_isle)->decisle, -1);//T_DG_RECALL로 빌드되야 한다.
		} else coupleisle->setFinal(1);
	}
	Dynagen(Impulse *imp, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, Isleloop *tar_zc_isle,
		bool pred_link = 1, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{
		dgeninst(imp, name);

		tar_zc_isle->setFinal(0);//encoder일 경우 이번에 후행이 연결되므로 최종이 아니게 되어 리셋한다.
		szlatent = latent_sz;
		dgindisc = indiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, ingate, tar_zc_isle->isleout, HID, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, ingate, tar_zc_isle->isleout, HID, (bytet *)"copule_isle");
		}
		tar_zc_isle->regiRouteTo(coupleisle, 1);//잠재코드 타겟값을 커플isle의 타겟값으로 주입
		if(pred_link) {//예측을 위한 연결, 커풀의 예측 출력값을 타겟isle의 디코더의 입력값으로 주입,이때 타겟isle은
			coupleisle->regiRouteTo(((Encodeisle *)tar_zc_isle)->decisle, -1);//T_DG_RECALL로 빌드되야 한다.
		} else coupleisle->setFinal(1);
		dgenimp->regifeed(coupleisle, 0);//입력값을 커플isle의 입력값으로 주입
	}
	Dynagen(Impulse *imp, Isleloop *in_zc_isle, Flux *targate, intt latent_sz, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx) //양측 자기구조 학습없이 바로 커플(연결)isle에 연결
	{
		dgeninst(imp, name);

		in_zc_isle->setFinal(0);//encoder일 경우 이번에 후행이 연결되므로 최종이 아니게 되어 리셋한다.
		szlatent = latent_sz;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, in_zc_isle->isleout, targate, HOD, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, in_zc_isle->isleout, targate, HOD, (bytet *)"copule_isle");
		}
		in_zc_isle->regiRouteTo(coupleisle, 0);//잠재코드 입력값을 커플isle의 입력값으로 주입
		dgenimp->regifeed(coupleisle, 1);//목표값을 커플isle의 목표값으로 주입
		coupleisle->setFinal(1);
	}
	Dynagen(Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{//양측 자기구조 학습없이 바로 커플(연결)isle에 연결, 기존 Coaxisle이나 Stratusisle를 다이나젠 망에서 사용할때.
		dgeninst(imp, name);

		szlatent = latent_sz;
		dgindisc = indiscret;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;
		lossIndice = 0;
		if(imp->impTcr->stratusOpt != 0) {
			coupleisle = new(dgenimp->impTcr) Stratusisle(dgenimp, this, ingate, targate, HAD, (bytet *)"copule_isle");
		} else {
			coupleisle = new(dgenimp->impTcr) Coaxisle(dgenimp, this, ingate, targate, HAD, (bytet *)"copule_isle");
		}
		dgenimp->regifeed(coupleisle, -2);//입력값과 타겟값을 커플isle의 입력과 타겟값으로 주입
		coupleisle->setFinal(1);
	}
	/*Dynagen(Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
		intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx)
	{//디버깅 테스트 단독 수행용 생성자.
		Isleloop *isl, *next;
		Isleloop *in_head_isle, *tar_head_isle;
		intt rsz;
		sytet tar_sid = 0;

		dgeninst(imp, name);

		szlatent = latent_sz;
		dgindisc = indiscret;
		dgoutdisc = outdiscret;
		szembede = embedim;
		actcode = af;
		rLeaning = lr;

		inz_bound = imp->impTcr->inzbound;
		tarz_bound = imp->impTcr->tarzbound;
		in_head_isle = tar_head_isle = nullx;
		rsz = inz_bound;
		lossIndice = 0;
		Encodeisle *ei = new(dgenimp->impTcr) Encodeisle(dgenimp, this, ingate, rsz, 1, HID, (bytet *)"copule_isle");

		intt dim[MX_DIM];
		memcpy(dim, ingate->fshape, 4 * sizeof(intt));
		dim[0] = 10;

		Flux *tf = flux(imp->impTcr, ingate->fdim, dim, tfloat, variable);
		tf->randn(0, 1);
		ei->islein[0]->feedf(tf);
		ei->findectrain = 1;//밑에서 디코더를 직접 실행할 것이므로
		while(1) {
			ei->trainloop(1);
			ei->predloop(1);
			ei->decisle->islein[0]->feedf(ei->isleout);
			ei->decisle->islein[1]->feedf(ei->islein[0]);
			//ei->decisle->islein[0]->printo(2, 10);
			//ei->decisle->islein[1]->printo(2, 10);
			ei->decisle->trainloop(1);
			dgenloss[lossIndice]->printo();
		}
	}*/
};
Impulse *impluse(Tracer *tcr, const bytet *name = nullx);
Dynagen *dynagen(sytet t_dynagen, Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret,
	intt outdiscret, intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Isleloop *in_zc_isle, Isleloop *tar_zc_isle, bool pred_link = 1,
	sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Flux *ingate, intt latent_sz, intt indiscret, intt embedim, Isleloop *tar_zc_isle,
	bool pred_link = 1, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Isleloop *in_zc_isle, Flux *targate, intt latent_sz, intt outdiscret,
	intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);
Dynagen *dynagen(Impulse *imp, Flux *ingate, Flux *targate, intt latent_sz, intt indiscret, intt outdiscret,
	intt embedim, sytet af = ACTF_TANH, floatt lr = -1, const bytet *name = nullx);